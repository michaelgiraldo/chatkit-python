import asyncio
import base64
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from typing import (
    Any,
    assert_never,
)
from uuid import uuid4

from pydantic import BaseModel

from chatkit.errors import (
    CustomStreamError,
    ErrorCode,
    InvalidOperationError,
    StreamError,
    UnsupportedOperationError,
)

from ._stream import _close_stream, _closing_stream
from .actions import Action
from .events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    ErrorEvent,
    StreamOptions,
    StreamOptionsEvent,
    SyncCustomActionResponse,
    ThreadCreatedEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    ThreadUpdatedEvent,
    WorkflowTaskAdded,
    WorkflowTaskUpdated,
)
from .logger import logger
from .models import (
    AssistantMessageContent,
    AssistantMessageItem,
    Attachment,
    AudioInput,
    ClientToolCallItem,
    FeedbackKind,
    Page,
    SDKHiddenContextItem,
    StructuredInputAnswer,
    StructuredInputItem,
    StructuredInputMultipleChoice,
    Thread,
    ThreadItem,
    ThreadMetadata,
    TranscriptionResult,
    UserMessageInput,
    UserMessageItem,
    WidgetItem,
    WorkflowItem,
)
from .requests import (
    AttachmentsCreateReq,
    AttachmentsDeleteReq,
    ChatKitReq,
    InputTranscribeReq,
    ItemsFeedbackReq,
    ItemsListReq,
    NonStreamingReq,
    StreamingReq,
    StructuredInputSubmission,
    ThreadsAddClientToolOutputReq,
    ThreadsAddStructuredInputReq,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsCustomActionReq,
    ThreadsDeleteReq,
    ThreadsGetByIdReq,
    ThreadsListReq,
    ThreadsRetryAfterItemReq,
    ThreadsSyncCustomActionReq,
    ThreadsUpdateReq,
    is_streaming_req,
)
from .store import (
    HIDDEN_ITEM_TYPES,
    AttachmentMetadataStore,
    AttachmentStore,
    ConflictError,
    NotFoundError,
    ThreadPatch,
    ThreadStore,
    TurnSupersededError,
)
from .turns import (
    ClientToolInvocation,
    ClientToolRequest,
    ClientToolTurn,
    PersistWorkflow,
    ResponderEvent,
    RetryTurn,
    StructuredInputTurn,
    Turn,
    TurnOutcome,
    TurnState,
    UserMessageTurn,
)

DEFAULT_PAGE_SIZE = 20
_TRANSIENT_OWNERSHIP_CHECK_INTERVAL = 0.25


def _stream_error_event(error: CustomStreamError | StreamError) -> ErrorEvent:
    if isinstance(error, CustomStreamError):
        return ErrorEvent(message=error.message, allow_retry=error.allow_retry)
    return ErrorEvent(code=error.code, allow_retry=error.allow_retry)


class StreamingResult(AsyncIterator[ThreadStreamEvent]):
    """A single-owner stream. Accepted turns expose their thread and turn IDs."""

    def __init__(
        self,
        stream: AsyncIterator[ThreadStreamEvent],
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        on_close_before_start: Callable[[], Awaitable[object]] | None = None,
    ):
        self.thread_id = thread_id
        self.turn_id = turn_id
        self._stream = stream
        self._on_close_before_start = on_close_before_start
        self._started = False
        self._closed = False

    async def __anext__(self) -> ThreadStreamEvent:
        if self._closed:
            raise StopAsyncIteration
        self._started = True
        return await anext(self._stream)

    async def aclose(self) -> None:
        """Close the stream and run its cleanup."""
        try:
            if not self._closed:
                self._closed = True
                await _close_stream(self._stream)
        finally:
            if not self._started and self._on_close_before_start is not None:
                await self._on_close_before_start()
                self._on_close_before_start = None


class EmptyResult(BaseModel):
    """An acknowledgement without response fields."""


type ImmediateResult = (
    Thread
    | Page[Thread]
    | Page[ThreadItem]
    | Attachment
    | TranscriptionResult
    | SyncCustomActionResponse
    | EmptyResult
)


class ChatKitServer[TContext = Any](ABC):
    def __init__(
        self,
        store: ThreadStore[TContext],
        attachment_store: AttachmentStore[TContext] | None = None,
        *,
        attachment_metadata_store: AttachmentMetadataStore[TContext] | None = None,
    ):
        """Configure thread persistence and optional attachment capabilities."""
        self.store = store
        self.attachment_store = attachment_store
        self.attachment_metadata_store = attachment_metadata_store

    def _require_client_tool_store(self) -> None:
        store_type = type(self.store)
        if (
            store_type.register_client_tool_call
            is ThreadStore.register_client_tool_call
            or store_type.load_client_tool_status is ThreadStore.load_client_tool_status
            or store_type.cancel_client_tool_call is ThreadStore.cancel_client_tool_call
        ):
            raise UnsupportedOperationError(
                "Client tool callbacks require a compatible thread store"
            )

    def _get_attachment_store(self) -> AttachmentStore[TContext]:
        """Return the configured AttachmentStore or raise if missing."""
        if self.attachment_store is None:
            raise UnsupportedOperationError(
                "AttachmentStore is not configured. Provide an AttachmentStore to ChatKitServer to handle file operations."
            )
        return self.attachment_store

    def _get_attachment_metadata_store(self) -> AttachmentMetadataStore[TContext]:
        if self.attachment_metadata_store is None:
            raise UnsupportedOperationError(
                "AttachmentMetadataStore is not configured. Pass "
                "attachment_metadata_store to ChatKitServer to use attachments."
            )
        return self.attachment_metadata_store

    @abstractmethod
    def respond(self, turn: Turn[TContext]) -> AsyncIterator[ResponderEvent]:
        """Stream events for a new message, a resumed turn, or a retry."""
        pass

    async def prepare_user_message(
        self, thread: ThreadMetadata, item: UserMessageItem, context: TContext
    ) -> UserMessageItem:
        """Validate or enrich a new user message before the server saves it.

        The thread snapshot and resolved attachments are available. For a new
        thread, the snapshot has an ID and has yet to be saved. Return the item
        with its original ID, thread ID, and creation time.

        Raise `InvalidOperationError` for a request error (HTTP 422 with the
        FastAPI adapter), or `CustomStreamError` for a user-facing error event.
        A rejected message produces no thread, attachment, or message writes.
        Retries and continuations use their stored inputs.
        """
        return item

    async def on_turn_finished(  # noqa: B027
        self, turn: Turn[TContext], outcome: TurnOutcome
    ) -> None:
        """Observe a turn's terminal outcome after server persistence or cleanup.

        A waiting turn has emitted a persisted pending tool call or structured input.
        Completion describes server processing, not delivery to a browser. Exceptions
        raised by this hook are logged.
        """
        pass

    async def update_thread(
        self,
        thread_id: str,
        patch: ThreadPatch,
        context: TContext,
        *,
        turn_id: str,
    ) -> ThreadUpdatedEvent:
        """Apply a patch as the active responder and return its browser event."""
        updated = await self.store.patch_thread(
            thread_id, patch, context, turn_id=turn_id
        )
        return ThreadUpdatedEvent(thread=self._to_thread_response(updated))

    async def add_feedback(  # noqa: B027
        self,
        thread_id: str,
        item_ids: list[str],
        feedback: FeedbackKind,
        context: TContext,
    ) -> None:
        """Persist user feedback for one or more thread items."""
        pass

    async def transcribe(  # noqa: B027
        self, audio_input: AudioInput, context: TContext
    ) -> TranscriptionResult:
        """Transcribe speech audio to text (dictation).

        Audio bytes are in `audio_input.data`. The raw MIME type is `audio_input.mime_type`; use
        `audio_input.media_type` for the base media type (one of: `audio/webm`, `audio/ogg`, `audio/mp4`).

        Args:
            audio_input: Audio bytes plus MIME type metadata for transcription.
            context: Arbitrary per-request context provided by the caller.

        Returns:
            A `TranscriptionResult` whose `text` is what should appear in the composer.
        """
        raise NotImplementedError(
            "transcribe() must be overridden to support the input.transcribe request."
        )

    def action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: TContext,
        *,
        turn_id: str,
    ) -> AsyncIterator[ResponderEvent]:
        """Handle a streaming action. Use its turn ID for direct store writes."""
        raise NotImplementedError(
            "The action() method must be overridden to react to actions. "
            "See https://michaelgiraldo.github.io/chatkit-python/guides/build-interactive-responses-with-widgets/#handle-widget-actions"
        )

    async def sync_action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: TContext,
    ) -> SyncCustomActionResponse:
        """Handle a widget or client-dispatched action and return a SyncCustomActionResponse."""
        raise NotImplementedError(
            "The sync_action() method must be overridden to react to sync actions. "
            "See https://michaelgiraldo.github.io/chatkit-python/guides/build-interactive-responses-with-widgets/#handle-widget-actions"
        )

    def get_stream_options(
        self, thread: ThreadMetadata, context: TContext
    ) -> StreamOptions:
        """
        Return stream-level runtime options. Allows the user to cancel the stream by default.
        Override this method to customize behavior.
        """
        return StreamOptions(allow_cancel=True)

    async def handle_stream_cancelled(
        self,
        thread: ThreadMetadata,
        pending_items: list[ThreadItem],
        context: TContext,
        *,
        turn_id: str,
    ):
        """Perform custom cleanup / stop inference when a stream is cancelled.
        Updates you make here will not be reflected in the UI until a reload.

        The default implementation persists any non-empty pending assistant messages
        to the thread but does not auto-save pending widget items or workflow items.

        Args:
            thread: The thread that was being processed.
            pending_items: Items that were not done streaming at cancellation time.
            context: Arbitrary per-request context provided by the caller.
            turn_id: The turn being cancelled. Use it to fence cleanup writes.
        """
        pending_assistant_message_items: list[AssistantMessageItem] = [
            item for item in pending_items if isinstance(item, AssistantMessageItem)
        ]
        for item in pending_assistant_message_items:
            is_empty = len(item.content) == 0 or all(
                (not content.text.strip()) for content in item.content
            )
            if not is_empty:
                await self.store.add_thread_item(
                    thread.id, item, context=context, turn_id=turn_id
                )

        # Add a hidden context item to the thread to indicate that the stream was cancelled.
        # Otherwise, depending on the timing of the cancellation, subsequent responses may
        # attempt to continue the cancelled response.
        await self.store.add_thread_item(
            thread.id,
            SDKHiddenContextItem(
                thread_id=thread.id,
                created_at=datetime.now(UTC),
                id=self.store.generate_item_id("sdk_hidden_context", thread, context),
                content="The user cancelled the stream. Stop responding to the prior request.",
            ),
            context=context,
            turn_id=turn_id,
        )

    async def process(
        self, request: ChatKitReq, context: TContext
    ) -> StreamingResult | ImmediateResult:
        """Process a validated request. Resolve streaming inputs before returning."""
        logger.info("Received request op: %s", request.type)
        if not is_streaming_req(request):
            return await self._process_non_streaming(request, context)
        try:
            return await self._prepare_streaming(request, context)
        except (CustomStreamError, StreamError) as error:
            event = _stream_error_event(error)

            async def rejection() -> AsyncIterator[ThreadStreamEvent]:
                yield event

            return StreamingResult(rejection())

    async def _process_non_streaming(
        self, request: NonStreamingReq, context: TContext
    ) -> ImmediateResult:
        match request:
            case ThreadsGetByIdReq():
                thread = await self._load_full_thread(
                    request.params.thread_id, context=context
                )
                return self._to_thread_response(thread)
            case ThreadsListReq():
                params = request.params
                threads = await self.store.load_threads(
                    limit=params.limit or DEFAULT_PAGE_SIZE,
                    after=params.after,
                    order=params.order,
                    context=context,
                )
                return Page(
                    has_more=threads.has_more,
                    after=threads.after,
                    data=[self._to_thread_response(thread) for thread in threads.data],
                )
            case ItemsFeedbackReq():
                await self.add_feedback(
                    request.params.thread_id,
                    request.params.item_ids,
                    request.params.kind,
                    context,
                )
                return EmptyResult()
            case AttachmentsCreateReq():
                attachment_store = self._get_attachment_store()
                metadata_store = self._get_attachment_metadata_store()
                attachment = await attachment_store.create_attachment(
                    request.params, context
                )
                await metadata_store.save_attachment(attachment, context=context)
                return attachment
            case AttachmentsDeleteReq():
                attachment_store = self._get_attachment_store()
                metadata_store = self._get_attachment_metadata_store()
                await attachment_store.delete_attachment(
                    request.params.attachment_id, context=context
                )
                await metadata_store.delete_attachment(
                    request.params.attachment_id, context=context
                )
                return EmptyResult()
            case InputTranscribeReq():
                if type(self).transcribe is ChatKitServer.transcribe:
                    raise UnsupportedOperationError("Transcription is unavailable.")
                try:
                    audio_bytes = base64.b64decode(
                        request.params.audio_base64, validate=True
                    )
                except ValueError as error:
                    raise InvalidOperationError(
                        "Audio must be Base64 encoded."
                    ) from error
                return await self.transcribe(
                    AudioInput(data=audio_bytes, mime_type=request.params.mime_type),
                    context=context,
                )
            case ItemsListReq():
                items_list_params = request.params
                return await self.store.load_thread_items(
                    items_list_params.thread_id,
                    limit=items_list_params.limit or DEFAULT_PAGE_SIZE,
                    order=items_list_params.order,
                    after=items_list_params.after,
                    context=context,
                    include_hidden=False,
                )
            case ThreadsUpdateReq():
                updated = await self.store.patch_thread(
                    request.params.thread_id, {"title": request.params.title}, context
                )
                return self._to_thread_response(updated)
            case ThreadsDeleteReq():
                await self.store.delete_thread(
                    request.params.thread_id, context=context
                )
                return EmptyResult()
            case ThreadsSyncCustomActionReq():
                return await self._process_sync_custom_action(request, context)
            case _:
                assert_never(request)

    async def _prepare_streaming(
        self, request: StreamingReq, context: TContext
    ) -> StreamingResult:
        turn_id = str(uuid4())
        match request:
            case ThreadsCreateReq():
                thread = ThreadMetadata(
                    id=self.store.generate_thread_id(context),
                    created_at=datetime.now(UTC),
                )
                user_message = await self._build_user_message_item(
                    request.params.input, thread, context
                )
                return await self._accept_user_message(
                    Turn(thread, UserMessageTurn(user_message), context, id=turn_id),
                    create_thread=True,
                )

            case ThreadsAddUserMessageReq():
                thread = await self.store.load_thread(
                    request.params.thread_id, context=context
                )
                user_message = await self._build_user_message_item(
                    request.params.input, thread, context
                )
                return await self._accept_user_message(
                    Turn(thread, UserMessageTurn(user_message), context, id=turn_id),
                )

            case ThreadsAddClientToolOutputReq():
                self._require_client_tool_store()
                thread = await self.store.load_thread(
                    request.params.thread_id, context=context
                )
                result = request.params.result
                status = await self.store.load_client_tool_status(
                    thread.id, result.item_id, context
                )
                if status != "pending":
                    raise ConflictError("The client tool call is no longer pending.")

                tool_call = await self.store.load_item(
                    thread.id, result.item_id, context=context
                )
                if (
                    not isinstance(tool_call, ClientToolCallItem)
                    or tool_call.id != result.item_id
                    or tool_call.thread_id != thread.id
                ):
                    raise InvalidOperationError(
                        "The requested item does not accept a client tool result."
                    )
                if tool_call.status != "pending":
                    raise ConflictError("The client tool call is no longer pending.")

                completed = tool_call.model_copy(
                    update={"output": result.output, "status": "completed"}
                )
                invocation = await self.store.claim_pending_item(
                    thread.id, tool_call, completed, context, next_turn_id=turn_id
                )
                if invocation is None:
                    raise ConflictError("The client tool has no executable invocation.")

                return self._respond_to_turn(
                    Turn(
                        thread,
                        ClientToolTurn(completed, invocation),
                        context,
                        id=turn_id,
                    )
                )

            case ThreadsAddStructuredInputReq():
                thread = await self.store.load_thread(
                    request.params.thread_id, context=context
                )
                item = await self.store.load_item(
                    request.params.thread_id,
                    request.params.item_id,
                    context=context,
                )
                if not isinstance(item, StructuredInputItem):
                    raise InvalidOperationError(
                        "The requested item does not accept structured input."
                    )

                updated_item = self._apply_structured_input_submission(
                    item,
                    request.params.input,
                )
                await self.store.claim_pending_item(
                    thread.id, item, updated_item, context, next_turn_id=turn_id
                )

                return self._respond_to_turn(
                    Turn(thread, StructuredInputTurn(updated_item), context, id=turn_id)
                )

            case ThreadsRetryAfterItemReq():
                thread_metadata = await self.store.load_thread(
                    request.params.thread_id, context=context
                )

                user_message_item = await self.store.truncate_after(
                    thread_metadata.id,
                    request.params.item_id,
                    context,
                    next_turn_id=turn_id,
                )
                return self._respond_to_turn(
                    Turn(
                        thread_metadata,
                        RetryTurn(user_message_item),
                        context,
                        id=turn_id,
                    )
                )

            case ThreadsCustomActionReq():
                if type(self).action is ChatKitServer.action:
                    raise UnsupportedOperationError(
                        "Streaming actions are unavailable."
                    )
                thread_metadata = await self.store.load_thread(
                    request.params.thread_id, context=context
                )

                item: ThreadItem | None = None
                if request.params.item_id:
                    item = await self.store.load_item(
                        request.params.thread_id,
                        request.params.item_id,
                        context=context,
                    )

                if item and not isinstance(item, WidgetItem):
                    raise InvalidOperationError(
                        "A custom action needs a widget sender."
                    )

                await self.store.begin_turn(thread_metadata.id, turn_id, context)
                return self._stream_turn(
                    thread_metadata,
                    turn_id,
                    context,
                    lambda: self.action(
                        thread_metadata,
                        request.params.action,
                        item,
                        context,
                        turn_id=turn_id,
                    ),
                )

            case _:
                assert_never(request)

    async def _process_sync_custom_action(
        self, request: ThreadsSyncCustomActionReq, context: TContext
    ) -> SyncCustomActionResponse:
        if type(self).sync_action is ChatKitServer.sync_action:
            raise UnsupportedOperationError("Synchronous actions are unavailable.")
        thread_metadata = await self.store.load_thread(
            request.params.thread_id, context=context
        )

        item: ThreadItem | None = None
        if request.params.item_id:
            item = await self.store.load_item(
                request.params.thread_id,
                request.params.item_id,
                context=context,
            )

        if item and not isinstance(item, WidgetItem):
            raise InvalidOperationError("A custom action needs a widget sender.")
        return await self.sync_action(
            thread_metadata,
            request.params.action,
            item,
            context,
        )

    def _apply_structured_input_submission(
        self,
        item: StructuredInputItem,
        submission: StructuredInputSubmission,
    ) -> StructuredInputItem:
        if item.status != "pending":
            raise ConflictError("The thread is not waiting for this form.")

        updated_item = item.model_copy(deep=True)
        questions_by_id = {question.id: question for question in updated_item.inputs}
        submitted_question_ids = set(submission.answers)
        unknown_question_ids = submitted_question_ids.difference(questions_by_id)
        if unknown_question_ids:
            unknown = ", ".join(sorted(unknown_question_ids))
            logger.warning(
                f"Structured input item {item.id} received unknown question id(s), ignoring: {unknown}"
            )

        for question in updated_item.inputs:
            answer = submission.answers.get(question.id)

            if (
                submission.status == "skipped"
                or answer is None
                or answer.skipped
                or not answer.values
            ):
                question.answer = StructuredInputAnswer(skipped=True)
                continue

            values = answer.values
            if isinstance(question, StructuredInputMultipleChoice):
                if not question.multiple:
                    values = values[:1]

            question.answer = StructuredInputAnswer(values=values)

        updated_item.status = submission.status
        return updated_item

    async def _accept_user_message(
        self,
        turn: Turn[TContext],
        *,
        create_thread: bool = False,
    ) -> StreamingResult:
        assert isinstance(turn.input, UserMessageTurn)
        item = turn.input.message
        for attachment in item.attachments:
            await self._get_attachment_metadata_store().save_attachment(
                attachment, context=turn.context
            )
        await self.store.begin_turn(
            turn.thread.id,
            turn.id,
            turn.context,
            message=item,
            create_thread=turn.thread if create_thread else None,
        )
        initial_events: list[ThreadStreamEvent] = []
        if create_thread:
            initial_events.append(
                ThreadCreatedEvent(thread=self._to_thread_response(turn.thread))
            )
        initial_events.append(ThreadItemDoneEvent(item=item))
        return self._respond_to_turn(turn, initial_events=initial_events)

    def _respond_to_turn(
        self,
        turn: Turn[TContext],
        *,
        initial_events: list[ThreadStreamEvent] | None = None,
    ) -> StreamingResult:
        # The store applied the replacement when it accepted the form.
        form_update = (
            ThreadItemReplacedEvent(item=turn.input.item)
            if isinstance(turn.input, StructuredInputTurn)
            else None
        )
        return self._stream_turn(
            turn.thread,
            turn.id,
            turn.context,
            lambda: self.respond(turn),
            turn=turn,
            initial_events=initial_events,
            form_update=form_update,
        )

    def _stream_turn(
        self,
        thread: ThreadMetadata,
        turn_id: str,
        context: TContext,
        stream: Callable[[], AsyncIterator[ResponderEvent]],
        *,
        turn: Turn[TContext] | None = None,
        initial_events: list[ThreadStreamEvent] | None = None,
        form_update: ThreadItemReplacedEvent | None = None,
    ) -> StreamingResult:
        return StreamingResult(
            self._run_turn(
                thread,
                turn_id,
                context,
                stream,
                turn=turn,
                initial_events=initial_events,
                form_update=form_update,
            ),
            thread_id=thread.id,
            turn_id=turn_id,
            on_close_before_start=lambda: self._finish_turn(
                thread.id, turn_id, TurnOutcome.CANCELLED, context
            ),
        )

    async def _finish_turn(
        self, thread_id: str, turn_id: str, outcome: TurnOutcome, context: TContext
    ) -> TurnState | None:
        try:
            return await self.store.finish_turn(thread_id, turn_id, outcome, context)
        except NotFoundError:
            # Deleting a thread also removes its turn records.
            return None

    async def _run_turn(
        self,
        thread: ThreadMetadata,
        turn_id: str,
        context: TContext,
        stream: Callable[[], AsyncIterator[ResponderEvent]],
        *,
        turn: Turn[TContext] | None,
        initial_events: list[ThreadStreamEvent] | None,
        form_update: ThreadItemReplacedEvent | None,
    ) -> AsyncIterator[ThreadStreamEvent]:
        outcome = TurnOutcome.COMPLETED
        pending_inputs: set[str] = set()
        started = False
        try:
            await self.store.start_turn(thread.id, turn_id, context)
            started = True
            for event in initial_events or ():
                await self.store.require_current_turn(thread.id, turn_id, context)
                yield event
            async with _closing_stream(
                self._process_events(
                    thread, turn_id, context, stream, form_update=form_update
                )
            ) as events:
                async for event in events:
                    match event:
                        case ErrorEvent():
                            outcome = TurnOutcome.FAILED
                        case (
                            ThreadItemDoneEvent(item=item)
                            | ThreadItemReplacedEvent(item=item)
                        ):
                            if (
                                isinstance(
                                    item, (ClientToolCallItem, StructuredInputItem)
                                )
                                and item.status == "pending"
                            ):
                                pending_inputs.add(item.id)
                            else:
                                pending_inputs.discard(item.id)
                        case ThreadItemRemovedEvent(item_id=item_id):
                            pending_inputs.discard(item_id)
                    yield event
            if outcome is TurnOutcome.COMPLETED and pending_inputs:
                outcome = TurnOutcome.WAITING
        except TurnSupersededError, NotFoundError:
            outcome = TurnOutcome.SUPERSEDED
        except asyncio.CancelledError, GeneratorExit:
            outcome = TurnOutcome.CANCELLED
            raise
        except Exception:
            outcome = TurnOutcome.FAILED
            raise
        finally:
            state = await self._finish_turn(thread.id, turn_id, outcome, context)
            if started and turn is not None:
                if state is None or state is TurnState.SUPERSEDED:
                    outcome = TurnOutcome.SUPERSEDED
                elif state is TurnState.WAITING:
                    outcome = TurnOutcome.WAITING
                try:
                    await self.on_turn_finished(turn, outcome)
                except Exception:
                    logger.exception("Error in on_turn_finished")

    async def _process_events(
        self,
        thread: ThreadMetadata,
        turn_id: str,
        context: TContext,
        stream: Callable[[], AsyncIterator[ResponderEvent]],
        *,
        form_update: ThreadItemReplacedEvent | None = None,
    ) -> AsyncIterator[ThreadStreamEvent]:
        # Send initial stream options
        await self.store.require_current_turn(thread.id, turn_id, context)
        yield StreamOptionsEvent(
            stream_options=self.get_stream_options(thread, context)
        )

        # Keep track of items that were streamed but not yet saved
        # so that we can persist them when the stream is cancelled.
        pending_items: dict[str, ThreadItem] = {}
        hidden_item_ids: set[str] = set()
        issued_tool_ids: set[str] = set()
        stream_failed = False
        loop = asyncio.get_running_loop()

        async def register_client_tool(request: ClientToolRequest) -> None:
            self._require_client_tool_store()
            invocation = ClientToolInvocation(
                item_id=request.item.id,
                thread_id=thread.id,
                turn_id=turn_id,
                call_id=request.call_id,
                model_continuation=request.model_continuation,
            )
            await self.store.register_client_tool_call(
                thread.id, request.item, invocation, context
            )
            issued_tool_ids.add(request.item.id)

        async def cancel_issued_tools() -> None:
            for item_id in issued_tool_ids.copy():
                await self.store.cancel_client_tool_call(
                    thread.id, item_id, context, turn_id=turn_id
                )
                issued_tool_ids.discard(item_id)

        try:
            if form_update is not None:
                await self.store.require_current_turn(thread.id, turn_id, context)
                yield form_update
            await self.store.require_current_turn(thread.id, turn_id, context)
            next_check = loop.time() + _TRANSIENT_OWNERSHIP_CHECK_INTERVAL
            async with _closing_stream(aiter(stream())) as events:
                while (event := await anext(events, None)) is not None:
                    if isinstance(event, ThreadItemAddedEvent):
                        # Stash an isolated copy in case we need to persist unfinished items
                        # on cancellation; downstream handlers keep using the original event.item.
                        pending_items[event.item.id] = event.item.model_copy(deep=True)

                    match event:
                        case (
                            ThreadItemAddedEvent(item=ClientToolCallItem())
                            | ThreadItemDoneEvent(item=ClientToolCallItem())
                            | ThreadItemReplacedEvent(item=ClientToolCallItem())
                        ):
                            raise ValueError(
                                "Publish client tools with ClientToolRequest"
                            )
                        case ClientToolRequest() as request:
                            if stream_failed:
                                continue
                            # A disconnect during a transaction must settle registration
                            # before cleanup decides which exact callback to cancel.
                            registration = asyncio.create_task(
                                register_client_tool(request)
                            )
                            try:
                                await asyncio.shield(registration)
                            except asyncio.CancelledError:
                                try:
                                    await registration
                                except TurnSupersededError:
                                    pass
                                except Exception:
                                    logger.exception("Error registering client tool")
                                raise
                            event = ThreadItemDoneEvent(item=request.item)
                        case ThreadItemDoneEvent():
                            await self.store.add_thread_item(
                                thread.id, event.item, context=context, turn_id=turn_id
                            )
                            pending_items.pop(event.item.id, None)
                        case ThreadItemRemovedEvent():
                            await self.store.delete_thread_item(
                                thread.id,
                                event.item_id,
                                context=context,
                                turn_id=turn_id,
                            )
                            pending_items.pop(event.item_id, None)
                            issued_tool_ids.discard(event.item_id)
                        case ThreadItemReplacedEvent():
                            await self.store.save_item(
                                thread.id, event.item, context=context, turn_id=turn_id
                            )
                            pending_items.pop(event.item.id, None)
                        case PersistWorkflow(item=item, replace=True):
                            await self.store.save_item(
                                thread.id, item, context=context, turn_id=turn_id
                            )
                            continue
                        case PersistWorkflow(item=item):
                            await self.store.add_thread_item(
                                thread.id, item, context=context, turn_id=turn_id
                            )
                            continue
                        case ThreadItemUpdatedEvent():
                            # Keep pending assistant message and workflow items up to date
                            # so that we have a reference to the latest version of these pending items
                            # when the stream is cancelled.
                            self._update_pending_items(pending_items, event)
                        case ErrorEvent():
                            stream_failed = True
                            await cancel_issued_tools()

                    if (
                        isinstance(
                            event,
                            (
                                ErrorEvent,
                                ThreadCreatedEvent,
                                ThreadUpdatedEvent,
                                ThreadItemDoneEvent,
                                ThreadItemRemovedEvent,
                                ThreadItemReplacedEvent,
                            ),
                        )
                        or loop.time() >= next_check
                    ):
                        await self.store.require_current_turn(
                            thread.id, turn_id, context
                        )
                        next_check = loop.time() + _TRANSIENT_OWNERSHIP_CHECK_INTERVAL
                    if isinstance(event, (ThreadCreatedEvent, ThreadUpdatedEvent)):
                        hidden_item_ids.update(
                            item.id
                            for item in event.thread.items.data
                            if item.type in HIDDEN_ITEM_TYPES
                        )
                        yield event.model_copy(
                            update={"thread": self._to_thread_response(event.thread)}
                        )
                    elif (
                        isinstance(
                            event,
                            (
                                ThreadItemAddedEvent,
                                ThreadItemDoneEvent,
                                ThreadItemReplacedEvent,
                            ),
                        )
                        and event.item.type in HIDDEN_ITEM_TYPES
                    ):
                        hidden_item_ids.add(event.item.id)
                    elif not (
                        isinstance(
                            event, (ThreadItemUpdatedEvent, ThreadItemRemovedEvent)
                        )
                        and event.item_id in hidden_item_ids
                    ):
                        yield event

        except asyncio.CancelledError, GeneratorExit:
            with suppress(TurnSupersededError):
                await self.store.require_current_turn(thread.id, turn_id, context)
                try:
                    await cancel_issued_tools()
                finally:
                    await self.handle_stream_cancelled(
                        thread, list(pending_items.values()), context, turn_id=turn_id
                    )
            raise
        except TurnSupersededError:
            raise
        except (CustomStreamError, StreamError) as error:
            await cancel_issued_tools()
            await self.store.require_current_turn(thread.id, turn_id, context)
            yield _stream_error_event(error)
        except Exception as e:
            await cancel_issued_tools()
            await self.store.require_current_turn(thread.id, turn_id, context)
            yield ErrorEvent(
                code=ErrorCode.STREAM_ERROR,
                allow_retry=True,
            )
            logger.exception(e)

    def _apply_assistant_message_update(
        self,
        item: AssistantMessageItem,
        update: AssistantMessageContentPartAdded
        | AssistantMessageContentPartTextDelta
        | AssistantMessageContentPartAnnotationAdded
        | AssistantMessageContentPartDone,
    ) -> AssistantMessageItem:
        # Pad the content list so the requested content_index exists before we write into it.
        # (Streaming updates can arrive for an index that hasn’t been created yet)
        while len(item.content) <= update.content_index:
            item.content.append(AssistantMessageContent(text="", annotations=[]))

        match update:
            case AssistantMessageContentPartAdded():
                item.content[update.content_index] = update.content
            case AssistantMessageContentPartTextDelta():
                item.content[update.content_index].text += update.delta
            case AssistantMessageContentPartAnnotationAdded():
                annotations = item.content[update.content_index].annotations
                if update.annotation_index <= len(annotations):
                    annotations.insert(update.annotation_index, update.annotation)
                else:
                    annotations.append(update.annotation)
            case AssistantMessageContentPartDone():
                item.content[update.content_index] = update.content
        return item

    def _update_pending_items(
        self,
        pending_items: dict[str, ThreadItem],
        event: ThreadItemUpdatedEvent,
    ):
        updated_item = pending_items.get(event.item_id)
        update = event.update
        match updated_item:
            case AssistantMessageItem():
                if isinstance(
                    update,
                    (
                        AssistantMessageContentPartAdded,
                        AssistantMessageContentPartTextDelta,
                        AssistantMessageContentPartAnnotationAdded,
                        AssistantMessageContentPartDone,
                    ),
                ):
                    pending_items[updated_item.id] = (
                        self._apply_assistant_message_update(updated_item, update)
                    )
            case WorkflowItem():
                if isinstance(update, (WorkflowTaskUpdated, WorkflowTaskAdded)):
                    match update:
                        case WorkflowTaskUpdated():
                            updated_item.workflow.tasks[update.task_index] = update.task
                        case WorkflowTaskAdded():
                            updated_item.workflow.tasks.append(update.task)

                    pending_items[updated_item.id] = updated_item
            case _:
                pass

    async def _build_user_message_item(
        self, input: UserMessageInput, thread: ThreadMetadata, context: TContext
    ) -> UserMessageItem:
        item = UserMessageItem(
            id=self.store.generate_item_id("message", thread, context),
            content=input.content,
            thread_id=thread.id,
            attachments=[
                (
                    await self._get_attachment_metadata_store().load_attachment(
                        attachment_id, context
                    )
                ).model_copy(update={"thread_id": thread.id})
                for attachment_id in input.attachments
            ],
            quoted_text=input.quoted_text,
            inference_options=input.inference_options,
            created_at=datetime.now(UTC),
        )
        identity = (item.id, item.thread_id, item.created_at)
        prepared = await self.prepare_user_message(thread, item, context)
        if (prepared.id, prepared.thread_id, prepared.created_at) != identity:
            raise ValueError("prepare_user_message must preserve message identity")
        return prepared

    async def _load_full_thread(self, thread_id: str, context: TContext) -> Thread:
        thread_meta = await self.store.load_thread(thread_id, context=context)
        thread_items = await self.store.load_thread_items(
            thread_id,
            after=None,
            limit=DEFAULT_PAGE_SIZE,
            order="asc",
            context=context,
            include_hidden=False,
        )
        return Thread(**thread_meta.model_dump(), items=thread_items)

    def _to_thread_response(self, thread: ThreadMetadata | Thread) -> Thread:
        items = (
            thread.items.model_copy(
                update={
                    "data": [
                        item
                        for item in thread.items.data
                        if item.type not in HIDDEN_ITEM_TYPES
                    ]
                }
            )
            if isinstance(thread, Thread)
            else Page()
        )

        return Thread(
            id=thread.id,
            title=thread.title,
            created_at=thread.created_at,
            items=items,
            status=thread.status,
            allowed_image_domains=thread.allowed_image_domains,
        )

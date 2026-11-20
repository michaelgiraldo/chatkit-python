import asyncio
import base64
import binascii
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    assert_never,
)

from pydantic import BaseModel

from chatkit.errors import (
    CustomStreamError,
    ErrorCode,
    InvalidOperationError,
    StreamError,
    UnsupportedOperationError,
)

from .actions import Action
from .logger import logger
from .store import (
    AttachmentMetadataStore,
    AttachmentStore,
    ConflictError,
    StoreItemType,
    ThreadPatch,
    ThreadStore,
    default_generate_id,
)
from .turns import (
    ClientToolTurn,
    RetryTurn,
    StructuredInputTurn,
    Turn,
    TurnOutcome,
    UserMessageTurn,
)
from .types import (
    AssistantMessageContent,
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    AssistantMessageItem,
    Attachment,
    AttachmentsCreateReq,
    AttachmentsDeleteReq,
    AudioInput,
    ChatKitReq,
    ClientToolCallItem,
    ErrorEvent,
    FeedbackKind,
    HiddenContextItem,
    InputTranscribeReq,
    ItemsFeedbackReq,
    ItemsListReq,
    NonStreamingReq,
    Page,
    SDKHiddenContextItem,
    StreamingReq,
    StreamOptions,
    StreamOptionsEvent,
    StructuredInputAnswer,
    StructuredInputItem,
    StructuredInputMultipleChoice,
    StructuredInputSubmission,
    SyncCustomActionResponse,
    Thread,
    ThreadCreatedEvent,
    ThreadItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
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
    ThreadStreamEvent,
    ThreadsUpdateReq,
    ThreadUpdatedEvent,
    TranscriptionResult,
    UserMessageInput,
    UserMessageItem,
    WidgetItem,
    WidgetRootUpdated,
    WidgetStreamingTextValueDelta,
    WorkflowItem,
    WorkflowTaskAdded,
    WorkflowTaskUpdated,
    is_streaming_req,
)
from .widgets import WidgetRoot, WidgetRootInput

DEFAULT_PAGE_SIZE = 20
DEFAULT_ERROR_MESSAGE = "An error occurred when generating a response."
_HIDDEN_ITEMS = (HiddenContextItem, SDKHiddenContextItem)


def _widget_state(widget: WidgetRootInput) -> WidgetRoot:
    return widget if isinstance(widget, WidgetRoot) else widget.build()


def diff_widget(
    before: WidgetRootInput, after: WidgetRootInput
) -> list[WidgetStreamingTextValueDelta | WidgetRootUpdated]:
    """Emit unambiguous text appends, or replace the widget for other changes."""
    before_payload = _widget_state(before).model_dump(mode="json")
    after_widget = _widget_state(after)
    after_payload = after_widget.model_dump(mode="json")

    def index_components(root: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        by_id: dict[str, list[dict[str, Any]]] = {}

        def visit(node: dict[str, Any]) -> None:
            if component_id := node.get("id"):
                by_id.setdefault(component_id, []).append(node)
            children = node.get("children")
            if isinstance(children, dict):
                visit(children)
            elif isinstance(children, list):
                for child in children:
                    visit(child)

        visit(root)
        return by_id

    before_index = index_components(before_payload)
    deltas: list[WidgetStreamingTextValueDelta | WidgetRootUpdated] = []
    for component_id, new_nodes in index_components(after_payload).items():
        old_nodes = before_index.get(component_id, [])
        if len(old_nodes) != 1 or len(new_nodes) != 1:
            continue
        old_node, new_node = old_nodes[0], new_nodes[0]
        old_value, new_value = old_node.get("value"), new_node.get("value")
        if (
            old_node["type"] in {"Text", "Markdown"}
            and old_node["type"] == new_node["type"]
            and isinstance(old_value, str)
            and isinstance(new_value, str)
            and new_value != old_value
            and new_value.startswith(old_value)
        ):
            old_node["value"] = new_value
            deltas.append(
                WidgetStreamingTextValueDelta(
                    component_id=component_id,
                    delta=new_value[len(old_value) :],
                    done=not new_node.get("streaming", False),
                )
            )

    # Once appends are applied, compare the remaining JSON, including extensions
    # and explicit nulls. JSON equality distinguishes boolean and numeric props.
    if json.dumps(before_payload, sort_keys=True) != json.dumps(
        after_payload, sort_keys=True
    ):
        return [WidgetRootUpdated(widget=after_widget)]
    return deltas


async def stream_widget(
    thread: ThreadMetadata,
    widget: WidgetRootInput | AsyncGenerator[WidgetRootInput, None],
    copy_text: str | None = None,
    generate_id: Callable[[StoreItemType], str] = default_generate_id,
) -> AsyncGenerator[ThreadStreamEvent, None]:
    """Stream a widget root or successive root states.

    An empty producer emits no events. Closing the returned stream also closes
    its producer.
    """

    if not isinstance(widget, AsyncGenerator):
        yield ThreadItemDoneEvent(
            item=WidgetItem(
                id=generate_id("message"),
                thread_id=thread.id,
                created_at=datetime.now(),
                widget=_widget_state(widget),
                copy_text=copy_text,
            ),
        )
        return

    async with _closing_stream(widget) as states:
        try:
            last_state = _widget_state(await anext(states)).model_copy(deep=True)
        except StopAsyncIteration:
            return

        item = WidgetItem(
            id=generate_id("message"),
            created_at=datetime.now(),
            widget=last_state,
            copy_text=copy_text,
            thread_id=thread.id,
        )
        yield ThreadItemAddedEvent(item=item)

        async for state in states:
            new_state = _widget_state(state).model_copy(deep=True)
            for update in diff_widget(last_state, new_state):
                yield ThreadItemUpdatedEvent(
                    item_id=item.id,
                    update=update,
                )
            last_state = new_state

    yield ThreadItemDoneEvent(
        item=item.model_copy(update={"widget": last_state}),
    )


async def _close_stream(stream: AsyncIterator[Any]) -> None:
    # The responder contract also accepts custom iterators that have no aclose().
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


@asynccontextmanager
async def _closing_stream[TEvent](
    stream: AsyncIterator[TEvent],
) -> AsyncIterator[AsyncIterator[TEvent]]:
    try:
        yield stream
    except BaseException:
        try:
            await _close_stream(stream)
        except Exception:
            logger.exception("Error while closing interrupted stream")
        raise
    else:
        await _close_stream(stream)


class StreamingResult(AsyncIterator[ThreadStreamEvent]):
    """A single-owner event stream. Close it when consumption ends early."""

    def __init__(self, stream: AsyncIterator[ThreadStreamEvent]):
        self._stream = stream

    async def __anext__(self) -> ThreadStreamEvent:
        return await anext(self._stream)

    async def aclose(self) -> None:
        """Close the stream and run its cleanup."""
        await _close_stream(self._stream)


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

    def _get_attachment_store(self) -> AttachmentStore[TContext]:
        """Return the configured AttachmentStore or raise if missing."""
        if self.attachment_store is None:
            raise UnsupportedOperationError(
                "AttachmentStore is not configured. Provide a AttachmentStore to ChatKitServer to handle file operations."
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
    def respond(self, turn: Turn[TContext]) -> AsyncIterator[ThreadStreamEvent]:
        """Stream events for a new message, a resumed turn, or a retry."""
        pass

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
        self, thread_id: str, patch: ThreadPatch, context: TContext
    ) -> ThreadUpdatedEvent:
        """Apply a patch and return an event that a responder can yield to the client."""
        updated = await self.store.patch_thread(thread_id, patch, context)
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
    ) -> AsyncIterator[ThreadStreamEvent]:
        """Handle a widget or client-dispatched action and yield response events."""
        raise NotImplementedError(
            "The action() method must be overridden to react to actions. "
            "See https://github.com/openai/chatkit-python/blob/main/docs/widgets.md#widget-actions"
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
            "See https://github.com/openai/chatkit-python/blob/main/docs/widgets.md#widget-actions"
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
    ):
        """Perform custom cleanup / stop inference when a stream is cancelled.
        Updates you make here will not be reflected in the UI until a reload.

        The default implementation persists any non-empty pending assistant messages
        to the thread but does not auto-save pending widget items or workflow items.

        Args:
            thread: The thread that was being processed.
            pending_items: Items that were not done streaming at cancellation time.
            context: Arbitrary per-request context provided by the caller.
        """
        pending_assistant_message_items: list[AssistantMessageItem] = [
            item for item in pending_items if isinstance(item, AssistantMessageItem)
        ]
        for item in pending_assistant_message_items:
            is_empty = len(item.content) == 0 or all(
                (not content.text.strip()) for content in item.content
            )
            if not is_empty:
                await self.store.add_thread_item(thread.id, item, context=context)

        # Add a hidden context item to the thread to indicate that the stream was cancelled.
        # Otherwise, depending on the timing of the cancellation, subsequent responses may
        # attempt to continue the cancelled response.
        await self.store.add_thread_item(
            thread.id,
            SDKHiddenContextItem(
                thread_id=thread.id,
                created_at=datetime.now(),
                id=self.store.generate_item_id("sdk_hidden_context", thread, context),
                content="The user cancelled the stream. Stop responding to the prior request.",
            ),
            context=context,
        )

    async def process(
        self, request: ChatKitReq, context: TContext
    ) -> StreamingResult | ImmediateResult:
        """Process a validated request. Resolve streaming inputs before returning."""
        logger.info("Received request op: %s", request.type)
        if is_streaming_req(request):
            return StreamingResult(await self._prepare_streaming(request, context))
        return await self._process_non_streaming(request, context)

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
                except (binascii.Error, ValueError) as error:
                    raise InvalidOperationError(
                        "Audio must be Base64 encoded."
                    ) from error
                return await self.transcribe(
                    AudioInput(data=audio_bytes, mime_type=request.params.mime_type),
                    context=context,
                )
            case ItemsListReq():
                items_list_params = request.params
                items = await self.store.load_thread_items(
                    items_list_params.thread_id,
                    limit=items_list_params.limit or DEFAULT_PAGE_SIZE,
                    order=items_list_params.order,
                    after=items_list_params.after,
                    context=context,
                )
                # filter out hidden context items
                items.data = [
                    item for item in items.data if not isinstance(item, _HIDDEN_ITEMS)
                ]
                return items
            case ThreadsUpdateReq():
                updated = await self.update_thread(
                    request.params.thread_id, {"title": request.params.title}, context
                )
                return updated.thread
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
    ) -> AsyncIterator[ThreadStreamEvent]:
        match request:
            case ThreadsCreateReq():
                thread = ThreadMetadata(
                    id=self.store.generate_thread_id(context),
                    created_at=datetime.now(),
                )
                user_message = await self._build_user_message_item(
                    request.params.input, thread, context
                )
                return self._process_new_thread_item_respond(
                    thread, user_message, context, create_thread=True
                )

            case ThreadsAddUserMessageReq():
                thread = await self.store.load_thread(
                    request.params.thread_id, context=context
                )
                user_message = await self._build_user_message_item(
                    request.params.input, thread, context
                )
                return self._process_new_thread_item_respond(
                    thread,
                    user_message,
                    context,
                )

            case ThreadsAddClientToolOutputReq():
                thread = await self.store.load_thread(
                    request.params.thread_id, context=context
                )
                items = await self.store.load_thread_items(
                    thread.id, None, 1, "desc", context
                )
                tool_call = next(
                    (
                        item
                        for item in items.data
                        if isinstance(item, ClientToolCallItem)
                        and item.status == "pending"
                    ),
                    None,
                )
                if not tool_call:
                    raise InvalidOperationError(
                        "The thread is not waiting for a client tool result."
                    )

                completed = tool_call.model_copy(
                    update={"output": request.params.result, "status": "completed"}
                )
                await self.store.claim_pending_item(
                    thread.id, tool_call, completed, context
                )

                return self._process_turn(
                    Turn(thread, ClientToolTurn(completed), context)
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
                    thread.id, item, updated_item, context
                )

                return self._process_turn(
                    Turn(thread, StructuredInputTurn(updated_item), context)
                )

            case ThreadsRetryAfterItemReq():
                thread_metadata = await self.store.load_thread(
                    request.params.thread_id, context=context
                )

                user_message_item, _ = await self.store.truncate_after(
                    thread_metadata.id, request.params.item_id, context
                )
                return self._process_turn(
                    Turn(thread_metadata, RetryTurn(user_message_item), context)
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

                return self._process_events(
                    thread_metadata,
                    context,
                    lambda: self.action(
                        thread_metadata,
                        request.params.action,
                        item,
                        context,
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

    async def _process_new_thread_item_respond(
        self,
        thread: ThreadMetadata,
        item: UserMessageItem,
        context: TContext,
        *,
        create_thread: bool = False,
    ) -> AsyncIterator[ThreadStreamEvent]:
        if create_thread:
            await self.store.save_thread(thread, context=context)
            yield ThreadCreatedEvent(thread=self._to_thread_response(thread))

        # Store the updated attachments (now that they have a thread id).
        for attachment in item.attachments:
            await self._get_attachment_metadata_store().save_attachment(
                attachment, context=context
            )

        await self.store.add_thread_item(thread.id, item, context=context)

        async with _closing_stream(
            self._process_turn(Turn(thread, UserMessageTurn(item), context))
        ) as events:
            async for event in events:
                yield event

    async def _process_turn(
        self, turn: Turn[TContext]
    ) -> AsyncIterator[ThreadStreamEvent]:
        async def respond_events() -> AsyncIterator[ThreadStreamEvent]:
            async with _closing_stream(aiter(self.respond(turn))) as events:
                async for event in events:
                    yield event

        outcome = TurnOutcome.COMPLETED
        pending_inputs: set[str] = set()
        try:
            if isinstance(turn.input, UserMessageTurn):
                # The message is already saved. Closing at this event still finishes the turn.
                yield ThreadItemDoneEvent(item=turn.input.message)
            # The store already applied this replacement when it claimed the form.
            form_update = (
                ThreadItemReplacedEvent(item=turn.input.item)
                if isinstance(turn.input, StructuredInputTurn)
                else None
            )
            async with _closing_stream(
                self._process_events(
                    turn.thread, turn.context, respond_events, form_update=form_update
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
        except asyncio.CancelledError, GeneratorExit:
            outcome = TurnOutcome.CANCELLED
            raise
        except Exception:
            outcome = TurnOutcome.FAILED
            raise
        finally:
            try:
                await self.on_turn_finished(turn, outcome)
            except Exception:
                logger.exception("Error in on_turn_finished")

    async def _process_events(
        self,
        thread: ThreadMetadata,
        context: TContext,
        stream: Callable[[], AsyncIterator[ThreadStreamEvent]],
        *,
        form_update: ThreadItemReplacedEvent | None = None,
    ) -> AsyncIterator[ThreadStreamEvent]:
        await asyncio.sleep(0)  # allow the response to start streaming

        # Send initial stream options
        yield StreamOptionsEvent(
            stream_options=self.get_stream_options(thread, context)
        )

        # Keep track of items that were streamed but not yet saved
        # so that we can persist them when the stream is cancelled.
        pending_items: dict[str, ThreadItem] = {}
        hidden_item_ids: set[str] = set()

        try:
            if form_update is not None:
                yield form_update
            async with _closing_stream(aiter(stream())) as events:
                while (event := await anext(events, None)) is not None:
                    if isinstance(event, ThreadItemAddedEvent):
                        # Stash an isolated copy in case we need to persist unfinished items
                        # on cancellation; downstream handlers keep using the original event.item.
                        pending_items[event.item.id] = event.item.model_copy(deep=True)

                    match event:
                        case ThreadItemDoneEvent():
                            await self.store.add_thread_item(
                                thread.id, event.item, context=context
                            )
                            pending_items.pop(event.item.id, None)
                        case ThreadItemRemovedEvent():
                            await self.store.delete_thread_item(
                                thread.id, event.item_id, context=context
                            )
                            pending_items.pop(event.item_id, None)
                        case ThreadItemReplacedEvent():
                            await self.store.save_item(
                                thread.id, event.item, context=context
                            )
                            pending_items.pop(event.item.id, None)
                        case ThreadItemUpdatedEvent():
                            # Keep pending assistant message and workflow items up to date
                            # so that we have a reference to the latest version of these pending items
                            # when the stream is cancelled.
                            self._update_pending_items(pending_items, event)

                    if isinstance(event, (ThreadCreatedEvent, ThreadUpdatedEvent)):
                        hidden_item_ids.update(
                            item.id
                            for item in event.thread.items.data
                            if isinstance(item, _HIDDEN_ITEMS)
                        )
                        yield event.model_copy(
                            update={"thread": self._to_thread_response(event.thread)}
                        )
                    elif isinstance(
                        event,
                        (
                            ThreadItemAddedEvent,
                            ThreadItemDoneEvent,
                            ThreadItemReplacedEvent,
                        ),
                    ) and isinstance(event.item, _HIDDEN_ITEMS):
                        hidden_item_ids.add(event.item.id)
                    elif not (
                        isinstance(
                            event, (ThreadItemUpdatedEvent, ThreadItemRemovedEvent)
                        )
                        and event.item_id in hidden_item_ids
                    ):
                        yield event

        except asyncio.CancelledError, GeneratorExit:
            await self.handle_stream_cancelled(
                thread, list(pending_items.values()), context
            )
            raise
        except CustomStreamError as e:
            yield ErrorEvent(
                code="custom",
                message=e.message,
                allow_retry=e.allow_retry,
            )
        except StreamError as e:
            yield ErrorEvent(
                code=e.code,
                allow_retry=e.allow_retry,
            )
        except Exception as e:
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
        return UserMessageItem(
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
            created_at=datetime.now(),
        )

    async def _load_full_thread(self, thread_id: str, context: TContext) -> Thread:
        thread_meta = await self.store.load_thread(thread_id, context=context)
        thread_items = await self.store.load_thread_items(
            thread_id,
            after=None,
            limit=DEFAULT_PAGE_SIZE,
            order="asc",
            context=context,
        )
        return Thread(**thread_meta.model_dump(), items=thread_items)

    def _to_thread_response(self, thread: ThreadMetadata | Thread) -> Thread:
        items = (
            thread.items.model_copy(
                update={
                    "data": [
                        item
                        for item in thread.items.data
                        if not isinstance(item, _HIDDEN_ITEMS)
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

import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import (
    Annotated,
    Any,
    Literal,
)

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunResultStreaming,
    StreamEvent,
    ToolCallOutputItem,
)
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_output_message import Content
from openai.types.responses.response_output_text import (
    Annotation as ResponsesAnnotation,
)
from openai.types.responses.response_output_text import (
    AnnotationContainerFileCitation,
    AnnotationFileCitation,
    AnnotationURLCitation,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    SkipValidation,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from ._stream import _close_stream
from ._web_search import _WebSearchProgress
from .events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    GeneratedImageUpdated,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    WorkflowTaskAdded,
    WorkflowTaskUpdated,
)
from .logger import logger
from .models import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageItem,
    AssistantMessagePhase,
    ClientToolCallItem,
    DurationSummary,
    FileSource,
    GeneratedImage,
    GeneratedImageItem,
    Task,
    ThoughtTask,
    ThreadMetadata,
    URLSource,
    Workflow,
    WorkflowItem,
    WorkflowSummary,
)
from .store import StoreItemType, ThreadStore
from .turns import (
    ClientToolLocalContinuation,
    ClientToolModelContinuation,
    ClientToolRequest,
    ClientToolResponsesContinuation,
    PersistWorkflow,
    ResponderEvent,
)
from .widget_stream import stream_widget
from .widgets import Markdown, Text, WidgetRootInput

_CHAT_COMPLETIONS_PLACEHOLDER_ID = "__fake_id__"
_CLIENT_TOOL_CONFLICT = "Only one client tool may be requested per turn."
_ANNOTATION_ADAPTER = TypeAdapter[ResponsesAnnotation](ResponsesAnnotation)
_CONTINUATION_ITEMS = TypeAdapter(list[dict[str, JsonValue]])


@dataclass(frozen=True, slots=True)
class _ClientToolRequest:
    item_id: str | None
    call_id: str
    name: str
    arguments: dict[str, Any]
    continuation: Literal["local", "responses"]


def _capture_model_continuation(
    request: _ClientToolRequest, result: RunResultStreaming
) -> ClientToolModelContinuation:
    if not result.is_complete or not result.raw_responses or result.interruptions:
        raise ValueError("The Agents SDK run did not finish on the browser tool call.")

    calls = [
        item
        for item in result.raw_responses[-1].output
        if isinstance(item, ResponseFunctionToolCall)
    ]
    call_ids = {item.call_id for item in calls}
    if request.call_id not in call_ids or len(call_ids) != len(calls):
        raise ValueError("The browser tool must stop the Agents SDK run at its call.")

    try:
        if request.continuation == "local":
            items = _CONTINUATION_ITEMS.validate_python(
                result.to_input_list(mode="normalized"), strict=True
            )
        else:
            items = _CONTINUATION_ITEMS.validate_python(
                [
                    item.to_input_item()
                    for item in result.new_items
                    if isinstance(item, ToolCallOutputItem) and item.call_id in call_ids
                ],
                strict=True,
            )
    except ValidationError:
        raise ValueError(
            "The Agents SDK did not provide JSON continuation items."
        ) from None

    output_positions: dict[str, list[int]] = defaultdict(list)
    for position, item in enumerate(items):
        if (
            item.get("type") == "function_call_output"
            and isinstance(call_id := item.get("call_id"), str)
            and call_id in call_ids
        ):
            if not isinstance(item.get("output"), (str, list)):
                raise ValueError("The Agents SDK provided an invalid function output.")
            output_positions[call_id].append(position)
    if any(len(output_positions[call_id]) != 1 for call_id in call_ids):
        raise ValueError(
            "The Agents SDK did not provide one output for each pending function call."
        )

    if request.continuation == "local":
        replayed = [
            position
            for position, item in enumerate(items)
            if item.get("type") == "function_call"
            and item.get("call_id") == request.call_id
        ]
        original = next(item for item in calls if item.call_id == request.call_id)
        if (
            len(replayed) != 1
            or replayed[0] >= output_positions[request.call_id][0]
            or any(
                items[replayed[0]].get(key) != value
                for key, value in {
                    "name": original.name,
                    "arguments": original.arguments,
                    "namespace": original.namespace,
                }.items()
            )
        ):
            raise ValueError(
                "The Agents SDK did not preserve the original browser function call."
            )
        return ClientToolLocalContinuation(items=items)

    response_id = result.last_response_id
    if not response_id or response_id == _CHAT_COMPLETIONS_PLACEHOLDER_ID:
        raise ValueError(
            "Responses continuation requires a real OpenAI Responses response ID."
        )
    if len(items) != len(call_ids):
        raise ValueError("Responses continuation may contain only function outputs.")
    return ClientToolResponsesContinuation(response_id=response_id, items=items)


class AgentContext[TContext](BaseModel):
    """State and app events for a single response run.

    Set `resume_from` to a saved, authorized workflow from this thread or to
    `None` for a fresh run. Construct the context before starting the runner.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thread: ThreadMetadata
    store: Annotated[ThreadStore[TContext], SkipValidation]
    request_context: TContext
    resume_from: WorkflowItem | None = Field(exclude=True, repr=False)
    _workflow_item: WorkflowItem | None = None
    _resumed_workflow_id: str | None = None
    _client_tool_request: _ClientToolRequest | None = None
    _client_tool_conflict: bool = False
    _events: asyncio.Queue[ThreadStreamEvent] = PrivateAttr(
        default_factory=asyncio.Queue
    )

    @model_validator(mode="before")
    @classmethod
    def _require_explicit_workflow(cls, data: Any) -> Any:
        if isinstance(data, dict) and "workflow_item" in data:
            raise ValueError("Replace workflow_item with resume_from.")
        return data

    def model_post_init(self, _context: Any, /) -> None:
        if self.resume_from is None:
            return
        if self.resume_from.thread_id != self.thread.id:
            raise ValueError("The saved workflow must belong to the current thread.")
        self._workflow_item = self.resume_from.model_copy(deep=True)
        self._resumed_workflow_id = self._workflow_item.id

    @property
    def workflow_item(self) -> WorkflowItem | None:
        """The active workflow for this run."""
        return self._workflow_item

    @workflow_item.setter
    def workflow_item(self, item: WorkflowItem | None) -> None:
        self._workflow_item = item

    def generate_id(
        self, type: StoreItemType, thread: ThreadMetadata | None = None
    ) -> str:
        """Generate a new store-backed id for the given item type."""
        if type == "thread":
            return self.store.generate_thread_id(self.request_context)
        return self.store.generate_item_id(
            type, thread or self.thread, self.request_context
        )

    def request_client_tool(
        self,
        tool: ToolContext[AgentContext[TContext]],
        *,
        name: str,
        arguments: dict[str, Any],
        continuation: Literal["local", "responses"],
    ) -> None:
        """Request a browser callback for this invocation of an Agents SDK tool.

        Configure the agent to stop at the calling tool. The browser receives one
        pending call when the run finishes and supplies its result on the next turn.
        Choose local replay or a continuation linked to a stored Responses response.
        """
        if tool.context is not self or not tool.tool_call_id:
            raise ValueError("Pass the ToolContext for the current tool invocation.")
        if self._client_tool_request is not None:
            self._client_tool_conflict = True
            raise ValueError(_CLIENT_TOOL_CONFLICT)
        if continuation not in ("local", "responses"):
            raise ValueError("Choose 'local' or 'responses' for model continuation.")

        self._client_tool_request = _ClientToolRequest(
            item_id=tool.tool_call.id if tool.tool_call is not None else None,
            call_id=tool.tool_call_id,
            name=name,
            arguments=arguments,
            continuation=continuation,
        )

    async def stream_widget(
        self,
        widget: WidgetRootInput | AsyncGenerator[WidgetRootInput],
        copy_text: str | None = None,
    ) -> None:
        """Stream a widget into the thread by enqueueing widget events."""
        async with aclosing(
            stream_widget(self.thread, widget, copy_text, self.generate_id)
        ) as events:
            async for event in events:
                await self.stream(event)

    async def end_workflow(
        self, summary: WorkflowSummary | None = None, expanded: bool = False
    ) -> None:
        """Finalize the active workflow item, optionally attaching a summary."""
        if self.workflow_item is not None:
            await self.stream(
                self._finish_workflow(
                    self.workflow_item, summary=summary, expanded=expanded
                )
            )

    def _finish_workflow(
        self,
        item: WorkflowItem,
        *,
        summary: WorkflowSummary | None = None,
        expanded: bool = False,
    ) -> ThreadItemDoneEvent | ThreadItemReplacedEvent:
        if self.workflow_item is not None and self.workflow_item.id == item.id:
            self.workflow_item = None

        if summary is not None:
            item.workflow.summary = summary
        elif item.workflow.summary is None:
            delta = datetime.now(item.created_at.tzinfo) - item.created_at
            item.workflow.summary = DurationSummary(duration=int(delta.total_seconds()))
        item.workflow.expanded = expanded
        if item.id == self._resumed_workflow_id:
            return ThreadItemReplacedEvent(item=item)
        return ThreadItemDoneEvent(item=item)

    async def start_workflow(self, workflow: Workflow) -> None:
        """Begin streaming a new workflow item."""
        item = self._open_workflow(workflow)

        if workflow.type != "reasoning" and not workflow.tasks:
            # Defer sending added event until we have tasks
            return

        await self.stream(ThreadItemAddedEvent(item=item))

    def _open_workflow(self, workflow: Workflow) -> WorkflowItem:
        item = WorkflowItem(
            id=self.generate_id("workflow"),
            created_at=datetime.now(UTC),
            workflow=workflow,
            thread_id=self.thread.id,
        )
        self.workflow_item = item
        return item

    async def update_workflow_task(self, task: Task, task_index: int) -> None:
        """Update an existing workflow task and stream the delta."""
        if self.workflow_item is None:
            raise ValueError("Workflow is not set")
        # ensure reference is updated in case task is a copy
        self.workflow_item.workflow.tasks[task_index] = task
        await self.stream(
            ThreadItemUpdatedEvent(
                item_id=self.workflow_item.id,
                update=WorkflowTaskUpdated(
                    task=task,
                    task_index=task_index,
                ),
            )
        )

    async def add_workflow_task(self, task: Task) -> None:
        """Append a workflow task and stream the appropriate event."""
        item = self.workflow_item or self._open_workflow(
            Workflow(type="custom", tasks=[])
        )
        workflow = item.workflow
        workflow.tasks.append(task)

        if (
            item.id != self._resumed_workflow_id
            and workflow.type != "reasoning"
            and len(workflow.tasks) == 1
        ):
            await self.stream(ThreadItemAddedEvent(item=item))
        else:
            await self.stream(
                ThreadItemUpdatedEvent(
                    item_id=item.id,
                    update=WorkflowTaskAdded(
                        task=task,
                        task_index=len(workflow.tasks) - 1,
                    ),
                )
            )

    async def stream(self, event: ThreadStreamEvent) -> None:
        """Enqueue an event. Raises QueueShutDown after the response finishes."""
        await self._events.put(event)


async def _merge_generators[T1, T2](
    a: AsyncIterator[T1],
    b: AsyncIterator[T2],
    *,
    on_abort: Callable[[], None] | None = None,
) -> AsyncGenerator[T1 | T2]:
    async def read(source: AsyncIterator[T1 | T2]) -> T1 | T2:
        return await anext(source)

    pending_tasks: dict[asyncio.Task[T1 | T2], AsyncIterator[T1 | T2]] = {
        asyncio.create_task(read(a), name="chatkit-agent-event"): a,
        asyncio.create_task(read(b), name="chatkit-context-event"): b,
    }
    try:
        while True:
            done, _ = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            exhausted = False
            for task in done:
                try:
                    event = task.result()
                except StopAsyncIteration:
                    exhausted = True
                else:
                    yield event
            if exhausted:
                # Keep events that arrived while we were yielding from the done set.
                for task in pending_tasks.keys() - done:
                    if not task.cancel():
                        try:
                            yield task.result()
                        except asyncio.CancelledError, StopAsyncIteration:
                            pass
                return

            for task in done:
                source = pending_tasks.pop(task)
                pending_tasks[
                    asyncio.create_task(read(source), name=task.get_name())
                ] = source
    except BaseException:
        if on_abort is not None:
            on_abort()
        raise
    finally:
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)


class _EventWrapper:
    def __init__(self, event: ThreadStreamEvent):
        self.event = event


class _AsyncQueueIterator(AsyncIterator[_EventWrapper]):
    def __init__(self, queue: asyncio.Queue[ThreadStreamEvent]) -> None:
        self._queue = queue

    async def __anext__(self) -> _EventWrapper:
        try:
            return _EventWrapper(await self._queue.get())
        except asyncio.QueueShutDown:
            raise StopAsyncIteration from None


@dataclass(slots=True)
class _StreamingThoughtTracker:
    item_id: str
    index: int
    task: ThoughtTask


class ResponseStreamConverter:
    """Used by `stream_agent_response` to convert streamed Agents SDK output
    into values used by ChatKit thread items and thread stream events.

    Defines overridable methods for adapting streamed data (such as image
    generation results and partial updates) into the forms expected by ChatKit.
    """

    partial_images: int | None = None
    """
    The expected number of partial image updates for an image generation result.

    When set, this value is used to normalize partial image indices into a
    progress value in the range [0, 1]. If unset, all partial image updates are
    assigned a progress value of 0.
    """

    def __init__(self, *, partial_images: int | None = None):
        """
        Args:
            partial_images: The expected number of partial image updates for image
                generation results, or None if no progress normalization should
                be performed.
        """
        self.partial_images = partial_images

    async def base64_image_to_url(
        self,
        image_id: str,
        base64_image: str,
        partial_image_index: int | None = None,
    ) -> str:
        """
        Convert a base64-encoded image into a URL.

        This method is used to produce the URL stored on thread items for image
        generation results.

        Args:
            image_id: The ID of the image generation call. This stays stable across partial image updates.
            base64_image: The base64-encoded image.
            partial_image_index: The index of the partial image update, starting from 0.

        Returns:
            A URL string.
        """
        return f"data:image/png;base64,{base64_image}"

    def partial_image_index_to_progress(self, partial_image_index: int) -> float:
        """
        Convert a partial image index into a normalized progress value.

        Args:
            partial_image_index: The index of the partial image update, starting from 0.

        Returns:
            A float between 0 and 1 representing progress for the image
            generation result.
        """
        if self.partial_images is None or self.partial_images <= 0:
            return 0.0

        return min(1.0, partial_image_index / self.partial_images)

    async def file_citation_to_annotation(
        self, file_citation: AnnotationFileCitation
    ) -> Annotation | None:
        """Convert a Responses API file citation into an assistant message annotation."""
        filename = file_citation.filename
        if not filename:
            return None

        return Annotation(
            source=FileSource(filename=filename, title=filename),
            index=file_citation.index,
        )

    async def container_file_citation_to_annotation(
        self, container_file_citation: AnnotationContainerFileCitation
    ) -> Annotation | None:
        """Convert a Responses API container file citation into an assistant message annotation."""
        filename = container_file_citation.filename
        if not filename:
            return None

        return Annotation(
            source=FileSource(filename=filename, title=filename),
            index=container_file_citation.end_index,
        )

    async def url_citation_to_annotation(
        self, url_citation: AnnotationURLCitation
    ) -> Annotation | None:
        """Convert a Responses API URL citation into an assistant message annotation."""
        return Annotation(
            source=URLSource(url=url_citation.url, title=url_citation.title),
            index=url_citation.end_index,
        )


_DEFAULT_RESPONSE_STREAM_CONVERTER = ResponseStreamConverter()


async def _convert_content(
    content: Content, converter: ResponseStreamConverter
) -> AssistantMessageContent:
    if content.type == "output_text":
        annotations = [
            await _convert_annotation(annotation, converter)
            for annotation in content.annotations
        ]
        annotations = [a for a in annotations if a is not None]
        return AssistantMessageContent(
            text=content.text,
            annotations=annotations,
        )
    else:
        return AssistantMessageContent(
            text=content.refusal,
            annotations=[],
        )


async def _convert_annotation(
    raw_annotation: object, converter: ResponseStreamConverter
) -> Annotation | None:
    if raw_annotation is None:
        return None

    # Stream events can use different model classes than completed response content.
    if isinstance(raw_annotation, BaseModel):
        raw_annotation = raw_annotation.model_dump()

    annotation = _ANNOTATION_ADAPTER.validate_python(raw_annotation)

    if annotation.type == "file_citation":
        return await converter.file_citation_to_annotation(annotation)

    if annotation.type == "url_citation":
        return await converter.url_citation_to_annotation(annotation)

    if annotation.type == "container_file_citation":
        return await converter.container_file_citation_to_annotation(annotation)

    return None


async def stream_agent_response(
    context: AgentContext,
    result: RunResultStreaming,
    *,
    converter: ResponseStreamConverter = _DEFAULT_RESPONSE_STREAM_CONVERTER,
) -> AsyncGenerator[ResponderEvent]:
    """
    Convert a streamed Agents SDK run into ChatKit thread stream events.

    This function consumes a streaming run result and yields `ThreadStreamEvent`
    objects and server-only persistence events as the run progresses.
    Closing the stream early cancels an active run.

    Args:
        context: The AgentContext to use for the stream.
        result: The RunResultStreaming to convert.
        converter: Defines overridable methods for adapting streamed data (such as image
            generation results and partial updates) into the forms expected by ChatKit.

    Returns:
        An async iterator that yields thread stream events representing the run result.
    """
    queue_iterator = _AsyncQueueIterator(context._events)
    model_events = aiter(result.stream_events())
    cancel_requested = False

    def cancel_run() -> None:
        nonlocal cancel_requested
        if cancel_requested or result.is_complete:
            return
        cancel_requested = True
        try:
            result.cancel()
        except Exception, asyncio.CancelledError:
            logger.exception("Failed to cancel the Agents SDK run")

    merged_events = _merge_generators(model_events, queue_iterator, on_abort=cancel_run)
    try:
        async with aclosing(
            _stream_agent_response(context, merged_events, queue_iterator, converter)
        ) as events:
            async for event in events:
                yield event
    except BaseException:
        cancel_run()
        context._events.shutdown(immediate=True)
        for stream in (merged_events, model_events):
            try:
                await _close_stream(stream)
            except Exception, asyncio.CancelledError:
                logger.exception("Failed to close an agent event stream")
        raise
    else:
        await _close_stream(model_events)
        if (request := context._client_tool_request) is not None:
            item_id = request.item_id
            if not item_id or item_id == _CHAT_COMPLETIONS_PLACEHOLDER_ID:
                item_id = context.generate_id("tool_call")
            yield ClientToolRequest(
                item=ClientToolCallItem(
                    id=item_id,
                    thread_id=context.thread.id,
                    name=request.name,
                    arguments=request.arguments,
                    created_at=datetime.now(UTC),
                ),
                call_id=request.call_id,
                model_continuation=_capture_model_continuation(request, result),
            )


async def _stream_agent_response(
    context: AgentContext,
    events: AsyncIterator[StreamEvent | _EventWrapper],
    queue_iterator: _AsyncQueueIterator,
    converter: ResponseStreamConverter,
) -> AsyncGenerator[ResponderEvent]:
    ctx = context
    thread = context.thread
    produced_items = set()
    synthetic_message_ids: dict[int, str] = {}
    pending_message_phases: dict[str, AssistantMessagePhase] = {}
    pending_images: dict[str, GeneratedImageItem] = {}
    streaming_thought: _StreamingThoughtTracker | None = None
    # item_id -> content_index -> annotation count
    item_annotation_count: defaultdict[str, defaultdict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    def message_id(item_id: str, output_index: int) -> str:
        # Agents reuses this placeholder in its Responses-shaped Chat Completions
        # events. Keep one store ID until the message at this index completes.
        if item_id != _CHAT_COMPLETIONS_PLACEHOLDER_ID:
            return item_id
        if output_index not in synthetic_message_ids:
            synthetic_message_ids[output_index] = ctx.generate_id("message")
        return synthetic_message_ids[output_index]

    def start_reasoning_workflow() -> WorkflowItem:
        workflow_item = ctx._open_workflow(Workflow(type="reasoning", tasks=[]))
        produced_items.add(workflow_item.id)
        return workflow_item

    search_progress = _WebSearchProgress(start_reasoning_workflow)

    def finish_workflow(item: WorkflowItem):
        search_progress.settle(item)
        return ctx._finish_workflow(item)

    def relay_context_event(event: ThreadStreamEvent) -> Iterator[ThreadStreamEvent]:
        if isinstance(event, (ThreadItemAddedEvent, ThreadItemDoneEvent)):
            item = event.item
            # End the active workflow when a different visual item is added.
            if (
                ctx.workflow_item
                and ctx.workflow_item.id != item.id
                and item.type not in {"client_tool_call", "hidden_context_item"}
            ):
                yield finish_workflow(ctx.workflow_item)

            if item.id != ctx._resumed_workflow_id:
                # Guardrails only remove items produced during this run.
                produced_items.add(item.id)
            if isinstance(event, ThreadItemAddedEvent) and isinstance(
                item, WorkflowItem
            ):
                ctx.workflow_item = item

        if isinstance(
            event, (ThreadItemDoneEvent, ThreadItemReplacedEvent)
        ) and isinstance(event.item, WorkflowItem):
            item = event.item
            if ctx.workflow_item is not None and ctx.workflow_item.id == item.id:
                ctx.workflow_item = None
            search_progress.settle(item)
            if (
                isinstance(event, ThreadItemDoneEvent)
                and item.id == ctx._resumed_workflow_id
            ):
                event = ThreadItemReplacedEvent(item=item)

        yield event

    try:
        async for event in events:
            # Events emitted from agent context helpers
            if isinstance(event, _EventWrapper):
                for context_event in relay_context_event(event.event):
                    yield context_event
                continue

            if event.type != "raw_response_event":
                # Ignore everything else that isn't a raw response event
                continue

            # Handle Responses events
            event = event.data
            search_events = search_progress.handle(event, ctx.workflow_item)
            if search_events is not None:
                for search_event in search_events:
                    yield search_event
                continue

            if event.type == "response.content_part.added":
                if event.part.type == "reasoning_text":
                    continue
                content = await _convert_content(event.part, converter)
                yield ThreadItemUpdatedEvent(
                    item_id=message_id(event.item_id, event.output_index),
                    update=AssistantMessageContentPartAdded(
                        content_index=event.content_index,
                        content=content,
                    ),
                )
            elif event.type == "response.output_text.delta":
                yield ThreadItemUpdatedEvent(
                    item_id=message_id(event.item_id, event.output_index),
                    update=AssistantMessageContentPartTextDelta(
                        content_index=event.content_index,
                        delta=event.delta,
                    ),
                )
            elif event.type == "response.output_text.done":
                yield ThreadItemUpdatedEvent(
                    item_id=message_id(event.item_id, event.output_index),
                    update=AssistantMessageContentPartDone(
                        content_index=event.content_index,
                        content=AssistantMessageContent(
                            text=event.text,
                            annotations=[],
                        ),
                    ),
                )
            elif event.type == "response.output_text.annotation.added":
                annotation = await _convert_annotation(event.annotation, converter)
                if annotation:
                    item_id = message_id(event.item_id, event.output_index)
                    # Manually track annotation indices per content part in case we drop an annotation that
                    # we can't convert to our internal representation (e.g. missing filename).
                    content_counts = item_annotation_count[item_id]
                    annotation_index = content_counts[event.content_index]
                    content_counts[event.content_index] += 1
                    yield ThreadItemUpdatedEvent(
                        item_id=item_id,
                        update=AssistantMessageContentPartAnnotationAdded(
                            content_index=event.content_index,
                            annotation_index=annotation_index,
                            annotation=annotation,
                        ),
                    )
                continue
            elif event.type == "response.output_item.added":
                item = event.item
                if item.type == "reasoning" and not ctx.workflow_item:
                    yield ThreadItemAddedEvent(item=start_reasoning_workflow())
                if item.type == "message":
                    if ctx.workflow_item:
                        yield finish_workflow(ctx.workflow_item)
                    item_id = message_id(item.id, event.output_index)
                    produced_items.add(item_id)
                    if item.phase is not None:
                        pending_message_phases[item_id] = item.phase
                    yield ThreadItemAddedEvent(
                        item=AssistantMessageItem(
                            id=item_id,
                            thread_id=thread.id,
                            phase=item.phase,
                            content=[
                                await _convert_content(c, converter)
                                for c in item.content
                            ],
                            created_at=datetime.now(UTC),
                        ),
                    )
                elif item.type == "image_generation_call":
                    image_item = GeneratedImageItem(
                        id=ctx.generate_id("message"),
                        thread_id=thread.id,
                        created_at=datetime.now(UTC),
                        image=None,
                    )
                    pending_images[item.id] = image_item
                    produced_items.add(image_item.id)
                    yield ThreadItemAddedEvent(item=image_item)
            elif event.type == "response.image_generation_call.partial_image":
                image_item = pending_images.get(event.item_id)
                if image_item is None:
                    continue

                url = await converter.base64_image_to_url(
                    image_id=event.item_id,
                    base64_image=event.partial_image_b64,
                    partial_image_index=event.partial_image_index,
                )
                progress = converter.partial_image_index_to_progress(
                    event.partial_image_index
                )

                yield ThreadItemUpdatedEvent(
                    item_id=image_item.id,
                    update=GeneratedImageUpdated(
                        image=GeneratedImage(id=event.item_id, url=url),
                        progress=progress,
                    ),
                )
            elif event.type == "response.reasoning_summary_text.delta":
                if not ctx.workflow_item:
                    continue

                # stream the first thought in a new workflow so that we can show it earlier
                if (
                    ctx.workflow_item.workflow.type == "reasoning"
                    and len(ctx.workflow_item.workflow.tasks) == 0
                ):
                    streaming_thought = _StreamingThoughtTracker(
                        item_id=event.item_id,
                        index=event.summary_index,
                        task=ThoughtTask(content=event.delta),
                    )
                    ctx.workflow_item.workflow.tasks.append(streaming_thought.task)
                    yield ThreadItemUpdatedEvent(
                        item_id=ctx.workflow_item.id,
                        update=WorkflowTaskAdded(
                            task=streaming_thought.task,
                            task_index=0,
                        ),
                    )
                elif (
                    streaming_thought
                    and streaming_thought.task in ctx.workflow_item.workflow.tasks
                    and event.item_id == streaming_thought.item_id
                    and event.summary_index == streaming_thought.index
                ):
                    streaming_thought.task.content += event.delta
                    yield ThreadItemUpdatedEvent(
                        item_id=ctx.workflow_item.id,
                        update=WorkflowTaskUpdated(
                            task=streaming_thought.task,
                            task_index=ctx.workflow_item.workflow.tasks.index(
                                streaming_thought.task
                            ),
                        ),
                    )
            elif event.type == "response.reasoning_summary_text.done":
                if ctx.workflow_item:
                    if (
                        streaming_thought
                        and streaming_thought.task in ctx.workflow_item.workflow.tasks
                        and event.item_id == streaming_thought.item_id
                        and event.summary_index == streaming_thought.index
                    ):
                        task = streaming_thought.task
                        task.content = event.text
                        streaming_thought = None
                        update = WorkflowTaskUpdated(
                            task=task,
                            task_index=ctx.workflow_item.workflow.tasks.index(task),
                        )
                    else:
                        task = ThoughtTask(content=event.text)
                        ctx.workflow_item.workflow.tasks.append(task)
                        update = WorkflowTaskAdded(
                            task=task,
                            task_index=len(ctx.workflow_item.workflow.tasks) - 1,
                        )
                    yield ThreadItemUpdatedEvent(
                        item_id=ctx.workflow_item.id,
                        update=update,
                    )
            elif event.type == "response.output_item.done":
                item = event.item
                if item.type == "message":
                    item_id = message_id(item.id, event.output_index)
                    initial_phase = pending_message_phases.pop(item_id, None)
                    if item.id == _CHAT_COMPLETIONS_PLACEHOLDER_ID:
                        del synthetic_message_ids[event.output_index]
                    produced_items.add(item_id)
                    yield ThreadItemDoneEvent(
                        item=AssistantMessageItem(
                            id=item_id,
                            thread_id=thread.id,
                            phase=item.phase or initial_phase,
                            content=[
                                await _convert_content(c, converter)
                                for c in item.content
                            ],
                            created_at=datetime.now(UTC),
                        ),
                    )
                elif item.type == "image_generation_call":
                    image_item = pending_images.pop(item.id, None)
                    if image_item is None or not item.result:
                        continue

                    url = await converter.base64_image_to_url(
                        image_id=item.id,
                        base64_image=item.result,
                    )
                    image = GeneratedImage(id=item.id, url=url)

                    yield ThreadItemDoneEvent(
                        item=image_item.model_copy(update={"image": image})
                    )

    except BaseException as error:
        if ctx.workflow_item and search_progress.has_tasks(ctx.workflow_item):
            ctx.workflow_item = None
        if isinstance(
            error, (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered)
        ):
            # Discard queued app events after a guardrail stops the provider.
            context._events.shutdown(immediate=True)
            for item_id in produced_items:
                yield ThreadItemRemovedEvent(item_id=item_id)
        raise

    if context._client_tool_conflict:
        # The Agents SDK can convert a tool exception into a tool result. Never
        # hand off an ambiguous request even if the run handled the exception.
        raise ValueError(_CLIENT_TOOL_CONFLICT)

    context._events.shutdown()

    # Settle native search state before yielding queued events. A disconnect
    # during the drain must not leave an active search in the agent context.
    finished_search_workflow = None
    if ctx.workflow_item and search_progress.has_tasks(ctx.workflow_item):
        finished_search_workflow = finish_workflow(ctx.workflow_item)

    # Drain remaining events
    async for event in queue_iterator:
        for context_event in relay_context_event(event.event):
            yield context_event

    if finished_search_workflow is not None:
        yield finished_search_workflow
    elif workflow := ctx.workflow_item:
        yield PersistWorkflow(
            item=workflow, replace=workflow.id == ctx._resumed_workflow_id
        )


async def accumulate_text[TWidget: Markdown | Text](
    events: AsyncIterator[StreamEvent],
    base_widget: TWidget,
) -> AsyncIterator[TWidget]:
    text = ""
    yield base_widget
    async for event in events:
        if event.type == "raw_response_event":
            if event.data.type == "response.output_text.delta":
                text += event.data.delta
                yield base_widget.model_copy(update={"value": text})
    yield base_widget.model_copy(update={"value": text, "streaming": False})

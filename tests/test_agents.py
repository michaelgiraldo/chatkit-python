import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    InputGuardrailTripwireTriggered,
    OutputGuardrail,
    OutputGuardrailResult,
    OutputGuardrailTripwireTriggered,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    RunResultStreaming,
    StreamEvent,
    ToolCallItem,
)
from agents.tool_context import ToolContext
from openai.types.responses import (
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseImageGenCallPartialImageEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningItem,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from openai.types.responses.response_file_search_tool_call import Result
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ActionSearchSource,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_output_text import (
    AnnotationContainerFileCitation as ResponsesAnnotationContainerFileCitation,
)
from openai.types.responses.response_output_text import (
    AnnotationFileCitation as ResponsesAnnotationFileCitation,
)
from openai.types.responses.response_output_text import (
    AnnotationFilePath as ResponsesAnnotationFilePath,
)
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponsesAnnotationURLCitation,
)
from openai.types.responses.response_output_text import (
    ResponseOutputText,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_text_done_event import ResponseTextDoneEvent

from chatkit.agents import (
    AgentContext,
    ResponseStreamConverter,
    accumulate_text,
    stream_agent_response,
)
from chatkit.events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    GeneratedImageUpdated,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    WorkflowTaskAdded,
    WorkflowTaskUpdated,
)
from chatkit.models import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageItem,
    CustomSummary,
    CustomTask,
    DurationSummary,
    FileSource,
    GeneratedImage,
    GeneratedImageItem,
    Page,
    SearchTask,
    ThoughtTask,
    Thread,
    URLSource,
    WidgetItem,
    Workflow,
    WorkflowItem,
)
from chatkit.widgets import Card, Text

thread = Thread(id="123", title="Test", created_at=datetime.now(), items=Page())

mock_store = Mock()
mock_store.generate_item_id = lambda item_type, thread, context: f"{item_type}_id"
mock_store.load_thread_items = AsyncMock(return_value=Page())
mock_store.add_thread_item = AsyncMock()


class RunResult(RunResultStreaming):
    """Provide the stream methods used by ChatKit with synthetic events."""

    def __init__(self) -> None:
        self.is_complete = False
        self._test_events: asyncio.Queue[StreamEvent | Exception | None] = (
            asyncio.Queue()
        )

    def add_event(self, event: StreamEvent) -> None:
        self._test_events.put_nowait(event)

    def done(self, error: Exception | None = None) -> None:
        self.is_complete = True
        self._test_events.put_nowait(error)

    def cancel(self, mode: Literal["immediate", "after_turn"] = "immediate") -> None:
        self.done()

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        while (event := await self._test_events.get()) is not None:
            if isinstance(event, Exception):
                raise event
            yield event

    def throw_input_guardrails(self) -> None:
        self.done(
            InputGuardrailTripwireTriggered(
                InputGuardrailResult(
                    guardrail=Mock(spec=InputGuardrail),
                    output=GuardrailFunctionOutput(
                        output_info=None,
                        tripwire_triggered=True,
                    ),
                )
            )
        )

    def throw_output_guardrails(self) -> None:
        self.done(
            OutputGuardrailTripwireTriggered(
                OutputGuardrailResult(
                    guardrail=Mock(spec=OutputGuardrail),
                    output=GuardrailFunctionOutput(
                        output_info=None,
                        tripwire_triggered=True,
                    ),
                    agent=Mock(spec=Agent),
                    agent_output=None,
                )
            )
        )


def make_result() -> RunResult:
    return RunResult()


async def all_events(
    events: AsyncIterator[ThreadStreamEvent],
) -> list[ThreadStreamEvent]:
    return [event async for event in events]


@pytest.mark.parametrize("second_title", ["Processing", "Next step"])
async def test_added_workflow_task_uses_appended_index(second_title):
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    await context.add_workflow_task(CustomTask(title="Processing"))
    await context.add_workflow_task(CustomTask(title=second_title))

    first_event = context._events.get_nowait()
    second_event = context._events.get_nowait()

    assert isinstance(first_event, ThreadItemAddedEvent)
    assert isinstance(second_event, ThreadItemUpdatedEvent)
    assert isinstance(second_event.update, WorkflowTaskAdded)
    assert second_event.update.task_index == 1
    assert context.workflow_item is not None
    assert second_event.update.task is context.workflow_item.workflow.tasks[1]


@pytest.mark.parametrize("second_text", ["Thinking", "Another thought"])
async def test_completed_reasoning_task_uses_appended_index(second_text):
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseReasoningItem(
                    id="reasoning_1", summary=[], type="reasoning"
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    for index, text in enumerate(["Thinking", second_text]):
        result.add_event(
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=Mock(
                    type="response.reasoning_summary_text.done",
                    item_id="reasoning_1",
                    summary_index=index,
                    text=text,
                ),
            )
        )
    result.done()

    events = await all_events(stream_agent_response(context, result))
    task_updates = [
        event.update
        for event in events
        if isinstance(event, ThreadItemUpdatedEvent)
        and isinstance(event.update, WorkflowTaskAdded)
    ]
    assert [update.task_index for update in task_updates] == [0, 1]
    assert [update.task for update in task_updates] == [
        ThoughtTask(content="Thinking"),
        ThoughtTask(content=second_text),
    ]


async def test_returns_widget_item():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    result.add_event(
        RunItemStreamEvent(name="tool_called", item=Mock(spec=ToolCallItem))
    )
    await context.stream_widget(Card(children=[Text(value="Hello, world!")]))
    result.done()

    events = await all_events(
        stream_agent_response(
            context=context,
            result=result,
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], ThreadItemDoneEvent)
    assert isinstance(events[0].item, WidgetItem)
    assert events[0].item.widget == Card(children=[Text(value="Hello, world!")]).build()


async def test_returns_widget_item_generator():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    result.add_event(
        RunItemStreamEvent(name="tool_called", item=Mock(spec=ToolCallItem))
    )

    def render_widget(i: int) -> Card:
        return Card(children=[Text(id="text", value="Hello, world"[:i])])

    async def widget_generator():
        yield render_widget(0)
        yield render_widget(12)

    await context.stream_widget(widget_generator())
    result.done()

    events = await all_events(
        stream_agent_response(
            context=context,
            result=result,
        )
    )

    assert len(events) == 3
    assert isinstance(events[0], ThreadItemAddedEvent)
    assert isinstance(events[0].item, WidgetItem)
    assert events[0].item.widget == Card(children=[Text(id="text", value="")]).build()

    assert isinstance(events[1], ThreadItemUpdatedEvent)
    assert events[1].update.type == "widget.streaming_text.value_delta"
    assert events[1].update.component_id == "text"
    assert events[1].update.delta == "Hello, world"

    assert isinstance(events[2], ThreadItemDoneEvent)
    assert isinstance(events[2].item, WidgetItem)
    assert (
        events[2].item.widget
        == Card(children=[Text(id="text", value="Hello, world")]).build()
    )


async def test_returns_widget_full_replace_generator():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    result.add_event(
        RunItemStreamEvent(name="tool_called", item=Mock(spec=ToolCallItem))
    )

    async def widget_generator():
        yield Card(children=[Text(id="text", value="Hello!")])
        yield Card(children=[Text(key="other text", value="World!", streaming=False)])

    await context.stream_widget(widget_generator())
    result.done()

    events = await all_events(
        stream_agent_response(
            context=context,
            result=result,
        )
    )

    assert len(events) == 3
    assert isinstance(events[0], ThreadItemAddedEvent)
    assert isinstance(events[0].item, WidgetItem)
    assert (
        events[0].item.widget
        == Card(children=[Text(id="text", value="Hello!")]).build()
    )

    assert isinstance(events[1], ThreadItemUpdatedEvent)
    assert events[1].update.type == "widget.root.updated"
    assert (
        events[1].update.widget
        == Card(
            children=[Text(key="other text", value="World!", streaming=False)]
        ).build()
    )

    assert isinstance(events[2], ThreadItemDoneEvent)
    assert isinstance(events[2].item, WidgetItem)
    assert (
        events[2].item.widget
        == Card(
            children=[Text(key="other text", value="World!", streaming=False)]
        ).build()
    )


async def test_duplicate_client_tool_requests_fail_the_handoff():
    context = AgentContext(
        thread=thread,
        store=mock_store,
        request_context=None,
    )

    def tool_context(index: int) -> ToolContext[AgentContext[None]]:
        return ToolContext(
            context=context,
            tool_name="get_selection",
            tool_arguments="{}",
            tool_call_id=f"call_{index}",
            tool_call=ResponseFunctionToolCall(
                id=f"fc_{index}",
                type="function_call",
                call_id=f"call_{index}",
                name="get_selection",
                arguments="{}",
                status="completed",
            ),
        )

    context.request_client_tool(tool_context(1), name="get_selection", arguments={})
    with pytest.raises(ValueError, match="Only one client tool"):
        context.request_client_tool(tool_context(2), name="get_selection", arguments={})

    # The SDK can turn a callback exception into a tool result and complete.
    result = make_result()
    result.done()
    with pytest.raises(ValueError, match="Only one client tool"):
        await all_events(stream_agent_response(context, result))


async def test_accumulate_text():
    def delta(text: str) -> RawResponsesStreamEvent:
        return RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseTextDeltaEvent(
                type="response.output_text.delta",
                delta=text,
                content_index=0,
                item_id="123",
                logprobs=[],
                output_index=0,
                sequence_number=0,
            ),
        )

    result = make_result()
    result.add_event(delta("Hello, "))
    result.add_event(delta("world!"))

    result.done()

    events = [
        event
        async for event in accumulate_text(
            result.stream_events(), Text(key="text", value="", streaming=True)
        )
    ]
    assert events == [
        Text(key="text", value="", streaming=True),
        Text(key="text", value="Hello, ", streaming=True),
        Text(key="text", value="Hello, world!", streaming=True),
        Text(key="text", value="Hello, world!", streaming=False),
    ]


async def test_stream_agent_response_yields_context_events_without_streaming_events():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    event = ThreadItemAddedEvent(
        item=WidgetItem(
            id="123",
            created_at=datetime.now(),
            thread_id=thread.id,
            widget=Card(children=[Text(id="text", value="Hello, world!")]).build(),
        ),
    )

    await context.stream(event)

    response_streamer = stream_agent_response(context, result)
    event = await response_streamer.__anext__()

    assert event.type == "thread.item.added"

    future = asyncio.ensure_future(response_streamer.__anext__())
    assert future.done() is False

    result.done()

    try:
        await future
        assert False, "expected StopAsyncIteration"
    except StopAsyncIteration:
        pass

    assert future.done() is True


async def test_stream_agent_response_maps_events():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    event = ThreadItemAddedEvent(
        item=WidgetItem(
            id="123",
            created_at=datetime.now(),
            thread_id=thread.id,
            widget=Card(children=[Text(id="text", value="Hello, world!")]).build(),
        ),
    )

    await context.stream(event)
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseTextDeltaEvent(
                type="response.output_text.delta",
                delta="Hello, world!",
                content_index=0,
                item_id="123",
                logprobs=[],
                output_index=0,
                sequence_number=0,
            ),
        )
    )

    response_streamer = stream_agent_response(context, result)
    event1 = await response_streamer.__anext__()
    event2 = await response_streamer.__anext__()

    assert {event1.type, event2.type} == {
        "thread.item.added",
        "thread.item.updated",
    }

    future = asyncio.ensure_future(response_streamer.__anext__())
    assert future.done() is False

    result.done()

    try:
        await future
        assert False, "expected StopAsyncIteration"
    except StopAsyncIteration:
        pass

    assert future.done() is True


async def test_chat_completions_allocates_a_new_id_when_an_output_index_is_reused():
    store = Mock(
        generate_item_id=Mock(side_effect=["message_1", "message_2"]),
        load_thread_items=AsyncMock(return_value=Page()),
    )
    context = AgentContext(thread=thread, store=store, request_context=None)
    result = make_result()
    placeholder = ResponseOutputMessage(
        id="__fake_id__",
        content=[],
        role="assistant",
        status="in_progress",
        type="message",
    )
    for sequence_number in (0, 2):
        result.add_event(
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=placeholder,
                    output_index=0,
                    sequence_number=sequence_number,
                ),
            )
        )
        result.add_event(
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    item=placeholder.model_copy(update={"status": "completed"}),
                    output_index=0,
                    sequence_number=sequence_number + 1,
                ),
            )
        )
    result.done()

    events = await all_events(stream_agent_response(context, result))
    assert [
        (event.type, event.item.id)
        for event in events
        if isinstance(event, (ThreadItemAddedEvent, ThreadItemDoneEvent))
    ] == [
        ("thread.item.added", "message_1"),
        ("thread.item.done", "message_1"),
        ("thread.item.added", "message_2"),
        ("thread.item.done", "message_2"),
    ]


@pytest.mark.parametrize(
    "raw_event,expected_event",
    [
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    delta="Hello, world!",
                    content_index=0,
                    item_id="123",
                    logprobs=[],
                    output_index=0,
                    sequence_number=0,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartTextDelta(
                    content_index=0,
                    delta="Hello, world!",
                ),
            ),
        ),
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    part=ResponseOutputText(
                        type="output_text",
                        text="New content",
                        annotations=[],
                    ),
                    content_index=1,
                    item_id="123",
                    output_index=0,
                    sequence_number=1,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartAdded(
                    content_index=1,
                    content=AssistantMessageContent(text="New content", annotations=[]),
                ),
            ),
        ),
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseTextDoneEvent(
                    type="response.output_text.done",
                    text="Final text",
                    content_index=0,
                    item_id="123",
                    logprobs=[],
                    output_index=0,
                    sequence_number=2,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartDone(
                    content_index=0,
                    content=AssistantMessageContent(
                        text="Final text",
                        annotations=[],
                    ),
                ),
            ),
        ),
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=Mock(
                    type="response.output_text.annotation.added",
                    annotation=ResponsesAnnotationFileCitation(
                        type="file_citation",
                        file_id="file_123",
                        filename="file.txt",
                        index=5,
                    ),
                    content_index=0,
                    item_id="123",
                    annotation_index=0,
                    output_index=0,
                    sequence_number=3,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartAnnotationAdded(
                    content_index=0,
                    annotation_index=0,
                    annotation=Annotation(
                        source=FileSource(filename="file.txt", title="file.txt"),
                        index=5,
                    ),
                ),
            ),
        ),
    ],
)
async def test_event_mapping(raw_event, expected_event):
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    result.add_event(raw_event)
    result.done()

    events = await all_events(stream_agent_response(context, result))
    if expected_event:
        assert events == [expected_event]
    else:
        assert events == []


async def test_stream_agent_response_emits_annotation_added_events():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    item_id = "item_123"

    def add_annotation_event(annotation, sequence_number):
        result.add_event(
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=Mock(
                    type="response.output_text.annotation.added",
                    annotation=annotation,
                    content_index=0,
                    item_id=item_id,
                    annotation_index=sequence_number,
                    output_index=0,
                    sequence_number=sequence_number,
                ),
            )
        )

    add_annotation_event(
        ResponsesAnnotationFileCitation(
            type="file_citation",
            file_id="file_invalid",
            filename="",
            index=0,
        ),
        sequence_number=0,
    )
    add_annotation_event(
        ResponsesAnnotationContainerFileCitation(
            type="container_file_citation",
            container_id="container_1",
            file_id="file_123",
            filename="container.txt",
            start_index=0,
            end_index=3,
        ),
        sequence_number=1,
    )
    add_annotation_event(
        ResponsesAnnotationURLCitation(
            type="url_citation",
            url="https://example.com",
            title="Example",
            start_index=1,
            end_index=5,
        ),
        sequence_number=2,
    )
    result.done()

    events = await all_events(stream_agent_response(context, result))
    assert events == [
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=0,
                annotation=Annotation(
                    source=FileSource(filename="container.txt", title="container.txt"),
                    index=3,
                ),
            ),
        ),
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=1,
                annotation=Annotation(
                    source=URLSource(
                        url="https://example.com",
                        title="Example",
                    ),
                    index=5,
                ),
            ),
        ),
    ]


async def test_stream_agent_response_annotation_added_normalizes_annotations():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    item_id = "item_123"

    def add_annotation_event(
        annotation: dict[str, object] | None, sequence_number: int
    ):
        event = ResponseOutputTextAnnotationAddedEvent.model_validate({
            "type": "response.output_text.annotation.added",
            "annotation": annotation,
            "content_index": 0,
            "item_id": item_id,
            "annotation_index": sequence_number,
            "output_index": 0,
            "sequence_number": sequence_number,
        })
        result.add_event(RawResponsesStreamEvent(type="raw_response_event", data=event))

    # Invalid file citation is dropped (empty filename)
    add_annotation_event(
        {
            "type": "file_citation",
            "file_id": "file_invalid",
            "filename": "",
            "index": 0,
        },
        sequence_number=0,
    )

    add_annotation_event(None, sequence_number=1)

    # The next citations still use indices 0 and 1 after the dropped events.
    add_annotation_event(
        {
            "type": "container_file_citation",
            "container_id": "container_1",
            "file_id": "file_123",
            "filename": "container.txt",
            "start_index": 0,
            "end_index": 3,
        },
        sequence_number=2,
    )
    add_annotation_event(
        {
            "type": "url_citation",
            "url": "https://example.com",
            "title": "Example",
            "start_index": 1,
            "end_index": 5,
        },
        sequence_number=3,
    )

    result.done()

    events = await all_events(stream_agent_response(context, result))
    assert events == [
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=0,
                annotation=Annotation(
                    source=FileSource(filename="container.txt", title="container.txt"),
                    index=3,
                ),
            ),
        ),
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=1,
                annotation=Annotation(
                    source=URLSource(url="https://example.com", title="Example"),
                    index=5,
                ),
            ),
        ),
    ]


async def test_custom_annotation_conversion_used_by_stream_agent_response():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    class CustomResponseStreamConverter(ResponseStreamConverter):
        def __init__(self):
            super().__init__()
            self.calls: list[str] = []

        async def file_citation_to_annotation(self, file_citation):
            self.calls.append("file_citation")
            return Annotation(
                source=FileSource(
                    filename="report.pdf",
                    title="Usage Report",
                    description="Usage report for the month of January",
                ),
                index=111,
            )

        async def url_citation_to_annotation(self, url_citation):
            self.calls.append("url_citation")
            return Annotation(
                source=URLSource(url="https://custom.example/url", title="Custom"),
                index=222,
            )

    converter = CustomResponseStreamConverter()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_text.annotation.added",
                annotation=ResponsesAnnotationFileCitation(
                    type="file_citation",
                    file_id="file_123",
                    filename="file.txt",
                    index=0,
                ),
                content_index=0,
                item_id="m_1",
                annotation_index=0,
                output_index=0,
                sequence_number=0,
            ),
        )
    )

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseOutputMessage(
                    id="m_1",
                    content=[
                        ResponseOutputText(
                            annotations=[
                                ResponsesAnnotationURLCitation(
                                    type="url_citation",
                                    url="https://example.com",
                                    title="Example",
                                    start_index=0,
                                    end_index=7,
                                )
                            ],
                            text="Hello!",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.done()

    events = await all_events(
        stream_agent_response(context, result, converter=converter)
    )
    assert len(events) == 2

    assert isinstance(events[0], ThreadItemUpdatedEvent)
    assert events[0].item_id == "m_1"
    assert events[0].update == AssistantMessageContentPartAnnotationAdded(
        content_index=0,
        annotation_index=0,
        annotation=Annotation(
            source=FileSource(
                filename="report.pdf",
                title="Usage Report",
                description="Usage report for the month of January",
            ),
            index=111,
        ),
    )

    assert isinstance(events[1], ThreadItemDoneEvent)
    assert isinstance(events[1].item, AssistantMessageItem)
    assert events[1].item.id == "m_1"
    assert events[1].item.content == [
        AssistantMessageContent(
            text="Hello!",
            annotations=[
                Annotation(
                    source=URLSource(url="https://custom.example/url", title="Custom"),
                    index=222,
                )
            ],
        )
    ]
    assert converter.calls == ["file_citation", "url_citation"]


@pytest.mark.parametrize("throw_guardrail", ["input", "output"])
async def test_stream_agent_response_yields_item_removed_event(throw_guardrail):
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseOutputMessage(
                    id="1",
                    content=[
                        ResponseOutputText(
                            annotations=[], type="output_text", text="Hello, world!"
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    await context.stream(
        ThreadItemAddedEvent(
            item=AssistantMessageItem(
                id="2",
                content=[AssistantMessageContent(text="Hello, world!")],
                thread_id=thread.id,
                created_at=datetime.now(),
            ),
        )
    )

    await context.stream(
        ThreadItemDoneEvent(
            item=WidgetItem(
                id="3",
                created_at=datetime.now(),
                thread_id=thread.id,
                widget=Card(children=[Text(id="text", value="Hello, world!")]).build(),
            ),
        )
    )

    iterator = stream_agent_response(context, result)

    n = 3
    events = []
    # Grab first 3 events to
    async for event in iterator:
        n -= 1
        events.append(event)
        if n == 0:
            break

    if throw_guardrail == "input":
        result.throw_input_guardrails()
    else:
        result.throw_output_guardrails()

    try:
        async for event in iterator:
            events.append(event)
        assert False, "Guardrail should have been thrown from stream_agent_response"
    except InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered:
        pass
    except Exception as e:
        assert False, f"Unexpected exception: {e}"

    deleted_item_ids = {
        event.item_id for event in events if event.type == "thread.item.removed"
    }
    assert deleted_item_ids == {"1", "2", "3"}


async def test_stream_agent_response_assistant_message_content_types():
    AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseFileSearchToolCall(
                    id="fs_0",
                    queries=["Hello, world!"],
                    status="completed",
                    type="file_search_call",
                    results=[
                        Result(
                            file_id="f_123",
                            filename="test.txt",
                            text="Hello, world!",
                            score=1.0,
                        ),
                        Result(
                            file_id="f_123",
                            filename="test.txt",
                            text="Hello, friends!",
                            score=0.5,
                        ),
                    ],
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseOutputMessage(
                    id="1",
                    content=[
                        ResponseOutputText(
                            annotations=[
                                ResponsesAnnotationFileCitation(
                                    type="file_citation",
                                    file_id="f_123",
                                    index=0,
                                    filename="test.txt",
                                ),
                                ResponsesAnnotationContainerFileCitation(
                                    type="container_file_citation",
                                    container_id="container_1",
                                    file_id="f_456",
                                    filename="container.txt",
                                    start_index=0,
                                    end_index=3,
                                ),
                                ResponsesAnnotationURLCitation(
                                    type="url_citation",
                                    url="https://www.google.com",
                                    title="Google",
                                    start_index=0,
                                    end_index=10,
                                ),
                                ResponsesAnnotationFilePath(
                                    type="file_path",
                                    file_id="123",
                                    index=0,
                                ),
                            ],
                            text="Hello, world!",
                            type="output_text",
                        ),
                        ResponseOutputText(
                            annotations=[],
                            text="Can't do that",
                            type="output_text",
                        ),
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )

    result.done()

    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    events = await all_events(stream_agent_response(context, result))
    assert len(events) == 1
    assert isinstance(events[0], ThreadItemDoneEvent)
    message = events[0].item
    assert isinstance(message, AssistantMessageItem)
    assert message.content == [
        AssistantMessageContent(
            annotations=[
                Annotation(
                    source=FileSource(
                        filename="test.txt",
                        title="test.txt",
                    ),
                    index=0,
                ),
                Annotation(
                    source=FileSource(
                        filename="container.txt",
                        title="container.txt",
                    ),
                    index=3,
                ),
                Annotation(
                    source=URLSource(
                        url="https://www.google.com",
                        title="Google",
                    ),
                    index=10,
                ),
            ],
            text="Hello, world!",
        ),
        AssistantMessageContent(text="Can't do that", annotations=[]),
    ]
    assert message.id == "1"


async def test_stream_agent_response_image_generation_events():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.added",
                item=Mock(type="image_generation_call", id="img_call_1"),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.done",
                item=Mock(
                    type="image_generation_call", id="img_call_1", result="dGVzdA=="
                ),
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.done()

    stream = stream_agent_response(context, result)
    event1 = await stream.__anext__()
    assert isinstance(event1, ThreadItemAddedEvent)
    assert isinstance(event1.item, GeneratedImageItem)
    assert event1.item.type == "generated_image"
    assert event1.item.id == "message_id"
    assert event1.item.image is None

    event2 = await stream.__anext__()
    assert isinstance(event2, ThreadItemDoneEvent)
    assert isinstance(event2.item, GeneratedImageItem)
    assert event2.item.id == event1.item.id
    assert event2.item.image == GeneratedImage(
        id="img_call_1", url="data:image/png;base64,dGVzdA=="
    )

    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_stream_agent_response_image_generation_events_with_custom_converter():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.added",
                item=Mock(type="image_generation_call", id="img_call_1"),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.done",
                item=Mock(
                    type="image_generation_call", id="img_call_1", result="dGVzdA=="
                ),
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.done()

    class CustomResponseStreamConverter(ResponseStreamConverter):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[str, str, int | None]] = []

        async def base64_image_to_url(
            self,
            image_id: str,
            base64_image: str,
            partial_image_index: int | None = None,
        ) -> str:
            self.calls.append((image_id, base64_image, partial_image_index))
            return f"https://example.com/{image_id}"

    converter = CustomResponseStreamConverter()
    stream = stream_agent_response(context, result, converter=converter)
    event1 = await stream.__anext__()
    assert isinstance(event1, ThreadItemAddedEvent)
    assert isinstance(event1.item, GeneratedImageItem)
    assert event1.item.image is None

    event2 = await stream.__anext__()
    assert isinstance(event2, ThreadItemDoneEvent)
    assert isinstance(event2.item, GeneratedImageItem)
    assert converter.calls == [("img_call_1", "dGVzdA==", None)]
    assert event2.item.image == GeneratedImage(
        id="img_call_1", url="https://example.com/img_call_1"
    )
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_stream_agent_response_image_generation_partial_progress():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.added",
                item=Mock(type="image_generation_call", id="img_call_1"),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.image_generation_call.partial_image",
                partial_image_b64="dGVzdA==",
                partial_image_index=1,
                item_id="img_call_1",
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.done",
                item=Mock(
                    type="image_generation_call", id="img_call_1", result="dGVzdA=="
                ),
                output_index=0,
                sequence_number=2,
            ),
        )
    )
    result.done()

    converter = ResponseStreamConverter(partial_images=3)
    events = await all_events(
        stream_agent_response(context, result, converter=converter)
    )

    assert len(events) == 3
    added_event, partial_event, done_event = events

    assert isinstance(added_event, ThreadItemAddedEvent)
    assert isinstance(added_event.item, GeneratedImageItem)

    assert isinstance(partial_event, ThreadItemUpdatedEvent)
    assert isinstance(partial_event.update, GeneratedImageUpdated)
    assert partial_event.update.progress == pytest.approx(1 / 3)
    assert partial_event.update.image == GeneratedImage(
        id="img_call_1", url="data:image/png;base64,dGVzdA=="
    )

    assert isinstance(done_event, ThreadItemDoneEvent)
    assert isinstance(done_event.item, GeneratedImageItem)
    assert done_event.item.image == GeneratedImage(
        id="img_call_1", url="data:image/png;base64,dGVzdA=="
    )


async def test_interleaved_image_generation_keeps_each_preview_and_result():
    store = Mock(
        generate_item_id=Mock(side_effect=["image_1", "image_2"]),
        load_thread_items=AsyncMock(return_value=Page()),
    )
    context = AgentContext(thread=thread, store=store, request_context=None)
    result = make_result()
    first = ImageGenerationCall(
        id="img_call_1", status="in_progress", type="image_generation_call"
    )
    second = ImageGenerationCall(
        id="img_call_2", status="in_progress", type="image_generation_call"
    )

    for event in [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=first,
            output_index=0,
            sequence_number=0,
        ),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=second,
            output_index=1,
            sequence_number=1,
        ),
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=first.id,
            output_index=0,
            partial_image_index=1,
            partial_image_b64="MQ==",
            sequence_number=2,
        ),
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=second.id,
            output_index=1,
            partial_image_index=1,
            partial_image_b64="Mg==",
            sequence_number=3,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=second.model_copy(update={"status": "completed", "result": "Mw=="}),
            output_index=1,
            sequence_number=4,
        ),
        # A late preview for the completed call must not update the pending call.
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=second.id,
            output_index=1,
            partial_image_index=2,
            partial_image_b64="MA==",
            sequence_number=5,
        ),
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=first.id,
            output_index=0,
            partial_image_index=2,
            partial_image_b64="NA==",
            sequence_number=6,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=first.model_copy(update={"status": "completed", "result": "NQ=="}),
            output_index=0,
            sequence_number=7,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=ResponseOutputMessage(
                id="assistant_message",
                type="message",
                role="assistant",
                status="completed",
                content=[],
            ),
            output_index=2,
            sequence_number=8,
        ),
    ]:
        result.add_event(RawResponsesStreamEvent(type="raw_response_event", data=event))
    result.done()

    events = await all_events(
        stream_agent_response(
            context, result, converter=ResponseStreamConverter(partial_images=3)
        )
    )
    assert [event.type for event in events] == [
        "thread.item.added",
        "thread.item.added",
        "thread.item.updated",
        "thread.item.updated",
        "thread.item.done",
        "thread.item.updated",
        "thread.item.done",
        "thread.item.done",
    ]
    for event, item_id in zip(events[:2], ("image_1", "image_2"), strict=True):
        assert isinstance(event, ThreadItemAddedEvent)
        assert isinstance(event.item, GeneratedImageItem)
        assert event.item.id == item_id
        assert event.item.image is None

    for index, item_id, provider_id, image_data, progress in (
        (2, "image_1", first.id, "MQ==", 1 / 3),
        (3, "image_2", second.id, "Mg==", 1 / 3),
        (5, "image_1", first.id, "NA==", 2 / 3),
    ):
        event = events[index]
        assert isinstance(event, ThreadItemUpdatedEvent)
        assert event.item_id == item_id
        assert isinstance(event.update, GeneratedImageUpdated)
        assert event.update.image == GeneratedImage(
            id=provider_id, url=f"data:image/png;base64,{image_data}"
        )
        assert event.update.progress == pytest.approx(progress)

    for index, item_id, provider_id, image_data in (
        (4, "image_2", second.id, "Mw=="),
        (6, "image_1", first.id, "NQ=="),
    ):
        event = events[index]
        assert isinstance(event, ThreadItemDoneEvent)
        assert isinstance(event.item, GeneratedImageItem)
        assert event.item.id == item_id
        assert event.item.image == GeneratedImage(
            id=provider_id, url=f"data:image/png;base64,{image_data}"
        )

    message = events[-1]
    assert isinstance(message, ThreadItemDoneEvent)
    assert isinstance(message.item, AssistantMessageItem)
    assert message.item.id == "assistant_message"


def _web_search_start(item_id: str, output_index: int) -> ResponseOutputItemAddedEvent:
    return ResponseOutputItemAddedEvent(
        type="response.output_item.added",
        item=ResponseFunctionWebSearch(
            type="web_search_call",
            id=item_id,
            status="in_progress",
            action=ActionSearch(type="search"),
        ),
        output_index=output_index,
        sequence_number=output_index,
    )


def _web_search_done(
    item_id: str, output_index: int, sequence_number: int, query: str, url: str
) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent(
        type="response.output_item.done",
        item=ResponseFunctionWebSearch(
            type="web_search_call",
            id=item_id,
            status="completed",
            action=ActionSearch(
                type="search",
                queries=[query],
                sources=[ActionSearchSource(type="url", url=url)],
            ),
        ),
        output_index=output_index,
        sequence_number=sequence_number,
    )


async def test_interleaved_web_searches_keep_their_task_indices_and_citations():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    first, second = "search_first", "search_second"
    first_url, second_url = "https://example.com/first", "https://example.com/second"
    citation = ResponsesAnnotationURLCitation(
        type="url_citation",
        title="Source two",
        url=second_url,
        start_index=0,
        end_index=6,
    )
    for event in (
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=ResponseReasoningItem(id="reasoning", summary=[], type="reasoning"),
            output_index=0,
            sequence_number=0,
        ),
        # Both starts have identical task data. Their provider IDs distinguish them.
        _web_search_start(first, 1),
        _web_search_start(second, 2),
        ResponseWebSearchCallInProgressEvent(
            type="response.web_search_call.in_progress",
            item_id=first,
            output_index=1,
            sequence_number=3,
        ),
        ResponseWebSearchCallSearchingEvent(
            type="response.web_search_call.searching",
            item_id=second,
            output_index=2,
            sequence_number=4,
        ),
        ResponseWebSearchCallCompletedEvent(
            type="response.web_search_call.completed",
            item_id=second,
            output_index=2,
            sequence_number=5,
        ),
        _web_search_done(second, 2, 6, "second query", second_url),
        # The first call is still active when this stale event for the second arrives.
        ResponseWebSearchCallSearchingEvent(
            type="response.web_search_call.searching",
            item_id=second,
            output_index=2,
            sequence_number=7,
        ),
        ResponseWebSearchCallSearchingEvent(
            type="response.web_search_call.searching",
            item_id=first,
            output_index=1,
            sequence_number=8,
        ),
        _web_search_done(first, 1, 9, "first query", first_url),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=ResponseOutputMessage(
                id="assistant_answer",
                type="message",
                role="assistant",
                status="in_progress",
                content=[],
            ),
            output_index=3,
            sequence_number=10,
        ),
        ResponseOutputTextAnnotationAddedEvent.model_validate({
            "type": "response.output_text.annotation.added",
            "annotation": citation.model_dump(),
            "content_index": 0,
            "item_id": "assistant_answer",
            "annotation_index": 0,
            "output_index": 3,
            "sequence_number": 11,
        }),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=ResponseOutputMessage(
                id="assistant_answer",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Answer", annotations=[citation]
                    )
                ],
            ),
            output_index=3,
            sequence_number=12,
        ),
    ):
        result.add_event(RawResponsesStreamEvent(data=event))
    result.done()

    events = await all_events(stream_agent_response(context, result))
    search_events = [
        event
        for event in events
        if isinstance(event, ThreadItemUpdatedEvent)
        and isinstance(event.update, (WorkflowTaskAdded, WorkflowTaskUpdated))
        and isinstance(event.update.task, SearchTask)
    ]
    assert [
        (event.update.task_index, event.update.task.status_indicator)
        for event in search_events
        if isinstance(event.update, WorkflowTaskAdded)
    ] == [(0, "loading"), (1, "loading")]
    update_states = []
    for event in search_events:
        if isinstance(event.update, WorkflowTaskUpdated):
            update_states.append((
                event.update.task_index,
                event.update.task.status_indicator,
            ))
    assert [index for index, state in update_states if state == "complete"][-1] == 0
    assert (1, "complete") in update_states
    assert (1, "loading") not in update_states[
        update_states.index((1, "complete")) + 1 :
    ]

    (workflow,) = [
        event.item
        for event in events
        if isinstance(event, ThreadItemDoneEvent)
        and isinstance(event.item, WorkflowItem)
    ]
    assert all(event.item_id == workflow.id for event in search_events)
    first_task, second_task = workflow.workflow.tasks
    for task, query, url in (
        (first_task, "first query", first_url),
        (second_task, "second query", second_url),
    ):
        assert isinstance(task, SearchTask)
        assert task.status_indicator == "complete"
        assert task.queries == [query]
        assert [source.url for source in task.sources] == [url]

    answers = [
        event.item
        for event in events
        if isinstance(event, ThreadItemDoneEvent)
        and isinstance(event.item, AssistantMessageItem)
    ]
    assert len(answers) == 1
    answer = answers[0]
    assert answer.id == "assistant_answer"
    assert answer.content[0].annotations == [
        Annotation(source=URLSource(url=second_url, title="Source two"), index=6)
    ]
    assert any(
        isinstance(event, ThreadItemUpdatedEvent)
        and event.item_id == answer.id
        and isinstance(event.update, AssistantMessageContentPartAnnotationAdded)
        and event.update.annotation == answer.content[0].annotations[0]
        for event in events
    )


@pytest.mark.parametrize("resume_workflow", [True, False])
async def test_active_web_search_at_end_of_stream_finishes_workflow(
    resume_workflow: bool,
):
    store = Mock(
        generate_item_id=Mock(return_value="search_workflow"),
        load_thread_items=AsyncMock(return_value=Page()),
        add_thread_item=AsyncMock(),
    )
    context = AgentContext(thread=thread, store=store, request_context=None)
    result = make_result()
    item_id = "search_incomplete"
    result.add_event(
        RawResponsesStreamEvent(
            data=ResponseWebSearchCallInProgressEvent(
                type="response.web_search_call.in_progress",
                item_id=item_id,
                output_index=0,
                sequence_number=0,
            )
        )
    )
    async with (
        asyncio.timeout(2),
        aclosing(
            stream_agent_response(context, result, resume_workflow=resume_workflow)
        ) as events,
    ):
        opened = await anext(events)
        assert isinstance(opened, ThreadItemAddedEvent)
        assert isinstance(opened.item, WorkflowItem)
        loading = await anext(events)
        assert isinstance(loading, ThreadItemUpdatedEvent)
        assert loading.update == WorkflowTaskAdded(
            task_index=0, task=SearchTask(status_indicator="loading")
        )

        result.add_event(
            RawResponsesStreamEvent(
                data=ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=ResponseFunctionWebSearch(
                        type="web_search_call",
                        id=item_id,
                        status="in_progress",
                        action=ActionSearch(type="search", query="partial search"),
                    ),
                    output_index=0,
                    sequence_number=1,
                )
            )
        )
        result.done()
        update, finished = await all_events(events)

    assert isinstance(update, ThreadItemUpdatedEvent)
    assert update.item_id == opened.item.id
    assert isinstance(update.update, WorkflowTaskUpdated)
    assert update.update.task_index == 0
    assert isinstance(finished, ThreadItemDoneEvent)
    assert isinstance(finished.item, WorkflowItem)
    assert finished.item.id == opened.item.id
    (task,) = finished.item.workflow.tasks
    assert isinstance(task, SearchTask)
    assert task.queries == ["partial search"]
    assert task.status_indicator == "none"
    assert context.workflow_item is None
    assert store.load_thread_items.await_count == int(resume_workflow)
    store.add_thread_item.assert_not_awaited()


async def test_web_search_finishes_resumed_workflow_with_replacement():
    thought = ThoughtTask(content="Existing reasoning")
    previous = WorkflowItem(
        id="saved_workflow",
        thread_id=thread.id,
        created_at=datetime.now(UTC),
        workflow=Workflow(type="reasoning", tasks=[thought]),
    )
    store = Mock(
        load_thread_items=AsyncMock(return_value=Page(data=[previous])),
        add_thread_item=AsyncMock(),
    )
    context = AgentContext(thread=thread, store=store, request_context=None)
    result = make_result()
    url = "https://example.com/resumed"
    result.add_event(
        RawResponsesStreamEvent(data=_web_search_start("search_resumed", 0))
    )
    result.add_event(
        RawResponsesStreamEvent(
            data=_web_search_done("search_resumed", 0, 1, "query", url)
        )
    )
    result.done()

    added, updated, replaced = await all_events(stream_agent_response(context, result))
    assert isinstance(added, ThreadItemUpdatedEvent)
    assert added.item_id == previous.id
    assert added.update == WorkflowTaskAdded(
        task_index=1, task=SearchTask(status_indicator="loading")
    )
    assert isinstance(updated, ThreadItemUpdatedEvent)
    assert isinstance(updated.update, WorkflowTaskUpdated)
    assert updated.item_id == previous.id
    assert updated.update.task_index == 1
    assert isinstance(replaced, ThreadItemReplacedEvent)
    assert isinstance(replaced.item, WorkflowItem)
    assert replaced.item.id == previous.id
    assert replaced.item.workflow.tasks == [thought, updated.update.task]
    assert isinstance(updated.update.task, SearchTask)
    assert updated.update.task.status_indicator == "complete"
    store.load_thread_items.assert_awaited_once_with(thread.id, None, 2, "desc", None)
    store.generate_item_id.assert_not_called()
    store.add_thread_item.assert_not_awaited()


@pytest.mark.parametrize("finish", ["explicit", "assistant_message", "remain_open"])
async def test_resumed_workflow_updates_existing_record(
    finish: Literal["explicit", "assistant_message", "remain_open"],
):
    previous = WorkflowItem(
        id="saved_workflow",
        thread_id=thread.id,
        created_at=datetime.now(UTC),
        workflow=Workflow(type="custom", tasks=[CustomTask(title="Previous step")]),
    )
    store = Mock(
        load_thread_items=AsyncMock(return_value=Page(data=[previous])),
        add_thread_item=AsyncMock(),
        save_item=AsyncMock(),
    )
    context = AgentContext(thread=thread, store=store, request_context=None)
    result = make_result()

    async with aclosing(stream_agent_response(context, result)) as stream:
        if finish == "explicit":
            # A reasoning update establishes that the relay loaded the saved workflow.
            result.add_event(
                RawResponsesStreamEvent(
                    data=Mock(
                        type="response.reasoning_summary_text.done",
                        item_id="reasoning",
                        summary_index=0,
                        text="Next step",
                    ),
                )
            )
            update = await anext(stream)
            assert isinstance(update, ThreadItemUpdatedEvent)
            assert update.item_id == previous.id
            await context.end_workflow(expanded=True)
        elif finish == "assistant_message":
            result.add_event(
                RawResponsesStreamEvent(
                    data=ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        item=ResponseOutputMessage(
                            id="assistant_reply",
                            content=[],
                            role="assistant",
                            status="in_progress",
                            type="message",
                        ),
                        output_index=0,
                        sequence_number=0,
                    ),
                )
            )
        result.done()
        events = await all_events(stream)

    store.add_thread_item.assert_not_awaited()
    if finish == "remain_open":
        assert events == []
        store.save_item.assert_awaited_once_with(thread.id, previous, None)
    else:
        assert isinstance(events[0], ThreadItemReplacedEvent)
        item = events[0].item
        assert isinstance(item, WorkflowItem)
        assert item.id == previous.id
        assert isinstance(item.workflow.summary, DurationSummary)
        assert item.workflow.expanded is (finish == "explicit")
        assert context.workflow_item is None
        store.save_item.assert_not_awaited()


async def test_prequeued_workflow_completion_does_not_reopen_workflow():
    store = Mock(
        generate_item_id=Mock(return_value="queued_workflow"),
        load_thread_items=AsyncMock(return_value=Page()),
        add_thread_item=AsyncMock(),
    )
    context = AgentContext(thread=thread, store=store, request_context=None)
    await context.start_workflow(
        Workflow(type="custom", tasks=[CustomTask(title="Queued step")])
    )
    await context.end_workflow()
    result = make_result()
    result.done()

    added, done = await all_events(stream_agent_response(context, result))
    assert isinstance(added, ThreadItemAddedEvent)
    assert isinstance(done, ThreadItemDoneEvent)
    assert added.item.id == done.item.id
    assert context.workflow_item is None
    store.add_thread_item.assert_not_awaited()
    with pytest.raises(asyncio.QueueShutDown):
        await context.stream(done)


async def test_prequeued_workflow_takes_precedence_over_saved_workflow():
    previous = WorkflowItem(
        id="saved_workflow",
        thread_id=thread.id,
        created_at=datetime.now(UTC),
        workflow=Workflow(type="custom", tasks=[CustomTask(title="Previous step")]),
    )
    store = Mock(
        generate_item_id=Mock(return_value="queued_workflow"),
        load_thread_items=AsyncMock(return_value=Page(data=[previous])),
        add_thread_item=AsyncMock(),
        save_item=AsyncMock(),
    )
    context = AgentContext(thread=thread, store=store, request_context=None)
    await context.start_workflow(
        Workflow(type="custom", tasks=[CustomTask(title="Queued step")])
    )
    result = make_result()
    result.done()

    (added,) = await all_events(stream_agent_response(context, result))
    assert isinstance(added, ThreadItemAddedEvent)
    assert added.item.id == "queued_workflow"
    assert context.workflow_item is added.item
    store.add_thread_item.assert_awaited_once_with(thread.id, added.item, None)
    store.save_item.assert_not_awaited()


async def test_cancelled_web_search_relay_closes_the_pending_model_read():
    store = Mock()
    store.generate_item_id.return_value = "workflow"
    store.load_thread_items = AsyncMock(return_value=Page())
    store.add_thread_item = AsyncMock()
    context = AgentContext(thread=thread, store=store, request_context=None)
    result = Mock(spec=RunResultStreaming)
    result.is_complete = False
    pending_read = asyncio.Event()
    runner_cancelled = asyncio.Event()
    source_closed = asyncio.Event()
    result.cancel.side_effect = runner_cancelled.set

    async def model_events() -> AsyncIterator[StreamEvent]:
        try:
            yield RawResponsesStreamEvent(data=_web_search_start("search_first", 0))
            yield RawResponsesStreamEvent(data=_web_search_start("search_second", 1))
            pending_read.set()
            yield await asyncio.Future[StreamEvent]()
        finally:
            await runner_cancelled.wait()
            source_closed.set()

    result.stream_events.side_effect = model_events
    async with (
        asyncio.timeout(2),
        aclosing(stream_agent_response(context, result)) as events,
    ):
        tasks: list[SearchTask] = []
        while len(tasks) < 2:
            event = await anext(events)
            if isinstance(event, ThreadItemAddedEvent) and isinstance(
                event.item, WorkflowItem
            ):
                tasks.extend(
                    task
                    for task in event.item.workflow.tasks
                    if isinstance(task, SearchTask)
                )
            elif (
                isinstance(event, ThreadItemUpdatedEvent)
                and isinstance(event.update, WorkflowTaskAdded)
                and isinstance(event.update.task, SearchTask)
            ):
                tasks.append(event.update.task)
        assert [task.status_indicator for task in tasks] == ["loading", "loading"]

        read = asyncio.create_task(anext(events))
        await pending_read.wait()
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
        with pytest.raises(StopAsyncIteration):
            await anext(events)

    result.cancel.assert_called_once_with()
    assert source_closed.is_set()
    assert context.workflow_item is None
    store.add_thread_item.assert_not_awaited()
    assert not any(
        task.get_name().startswith("chatkit-") for task in asyncio.all_tasks()
    )


async def test_workflow_streams_first_thought():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    # first thought
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseReasoningItem(
                    id="resp_1",
                    summary=[],
                    type="reasoning",
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.reasoning_summary_text.delta",
                item_id="resp_1",
                summary_index=0,
                delta="Think",
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.reasoning_summary_text.delta",
                item_id="resp_1",
                summary_index=0,
                delta="ing 1",
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.reasoning_summary_text.done",
                item_id="resp_1",
                summary_index=0,
                text="Thinking 1",
            ),
        )
    )

    # second thought
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.reasoning_summary_text.delta",
                item_id="resp_1",
                summary_index=1,
                delta="Think",
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.reasoning_summary_text.delta",
                item_id="resp_1",
                summary_index=1,
                delta="ing 2",
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.reasoning_summary_text.done",
                item_id="resp_1",
                summary_index=1,
                text="Thinking 2",
            ),
        )
    )

    result.done()
    stream = stream_agent_response(context, result)

    # Workflow added
    event = await anext(stream)
    assert isinstance(event, ThreadItemAddedEvent)
    assert context.workflow_item is not None
    assert context.workflow_item.workflow.type == "reasoning"
    assert len(context.workflow_item.workflow.tasks) == 0
    assert event == ThreadItemAddedEvent(item=context.workflow_item)

    # First thought added
    event = await anext(stream)
    assert context.workflow_item is not None
    assert len(context.workflow_item.workflow.tasks) == 1
    assert isinstance(event, ThreadItemUpdatedEvent)
    assert event == ThreadItemUpdatedEvent(
        item_id=context.workflow_item.id,
        update=WorkflowTaskAdded(
            task=ThoughtTask(content="Think"),
            task_index=0,
        ),
    )

    # First thought delta
    event = await anext(stream)
    assert context.workflow_item is not None
    assert len(context.workflow_item.workflow.tasks) == 1
    assert isinstance(event, ThreadItemUpdatedEvent)
    assert event == ThreadItemUpdatedEvent(
        item_id=context.workflow_item.id,
        update=WorkflowTaskUpdated(
            task=ThoughtTask(content="Thinking 1"),
            task_index=0,
        ),
    )

    # First thought done
    event = await anext(stream)
    assert context.workflow_item is not None
    assert len(context.workflow_item.workflow.tasks) == 1
    assert isinstance(event, ThreadItemUpdatedEvent)
    assert event == ThreadItemUpdatedEvent(
        item_id=context.workflow_item.id,
        update=WorkflowTaskUpdated(
            task=ThoughtTask(content="Thinking 1"),
            task_index=0,
        ),
    )

    # Second thought added (not streamed)
    event = await anext(stream)
    assert context.workflow_item is not None
    assert len(context.workflow_item.workflow.tasks) == 2
    assert isinstance(event, ThreadItemUpdatedEvent)
    assert event == ThreadItemUpdatedEvent(
        item_id=context.workflow_item.id,
        update=WorkflowTaskAdded(
            task=ThoughtTask(content="Thinking 2"),
            task_index=1,
        ),
    )

    try:
        while True:
            await anext(stream)
    except StopAsyncIteration:
        pass


async def test_workflow_ends_on_message():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()

    # first thought
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseReasoningItem(
                    id="resp_1",
                    summary=[],
                    type="reasoning",
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.reasoning_summary_text.done",
                item_id="resp_1",
                summary_index=0,
                text="Thinking 1",
            ),
        )
    )

    # not reasoning
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseOutputMessage(
                    id="m_1",
                    content=[],
                    role="assistant",
                    status="in_progress",
                    type="message",
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )

    result.done()
    stream = stream_agent_response(context, result)

    # Workflow added
    event = await anext(stream)
    assert isinstance(event, ThreadItemAddedEvent)
    assert context.workflow_item is not None
    assert context.workflow_item.workflow.type == "reasoning"
    assert len(context.workflow_item.workflow.tasks) == 0
    assert event == ThreadItemAddedEvent(item=context.workflow_item)

    # First thought done
    event = await anext(stream)
    assert context.workflow_item is not None
    assert len(context.workflow_item.workflow.tasks) == 1
    assert isinstance(event, ThreadItemUpdatedEvent)
    assert event == ThreadItemUpdatedEvent(
        item_id=context.workflow_item.id,
        update=WorkflowTaskAdded(
            task=ThoughtTask(content="Thinking 1"),
            task_index=0,
        ),
    )

    # Workflow ended
    event = await anext(stream)
    assert isinstance(event, ThreadItemDoneEvent)
    assert event.item.type == "workflow"
    assert context.workflow_item is None
    # Summary and expanded are handled by the end_workflow method
    assert isinstance(event.item.workflow.summary, DurationSummary)
    assert event.item.workflow.expanded is False

    try:
        while True:
            await anext(stream)
    except StopAsyncIteration:
        pass


async def test_existing_workflow_summary_not_overwritten_on_automatic_end():
    context = AgentContext(thread=thread, store=mock_store, request_context=None)
    result = make_result()
    context.workflow_item = WorkflowItem(
        id="wf_1",
        created_at=datetime.now(),
        workflow=Workflow(type="custom", tasks=[], summary=CustomSummary(title="Test")),
        thread_id=thread.id,
    )

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                item=ResponseOutputMessage(
                    id="m_1",
                    content=[],
                    role="assistant",
                    status="in_progress",
                    type="message",
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )

    result.done()
    stream = stream_agent_response(context, result)

    event = await anext(stream)

    assert isinstance(event, ThreadItemDoneEvent)
    assert context.workflow_item is None
    assert event.item.type == "workflow"
    assert event.item.workflow.summary == CustomSummary(title="Test")

    try:
        while True:
            await anext(stream)
    except StopAsyncIteration:
        pass

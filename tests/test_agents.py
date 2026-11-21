import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock

import pytest
from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    ModelResponse,
    OutputGuardrailTripwireTriggered,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    RunResultStreaming,
    StreamEvent,
    ToolCallItem,
    ToolCallOutputItem,
    Usage,
)
from agents.tool_context import ToolContext
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningItem,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ActionSearchSource,
)
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponsesAnnotationURLCitation,
)
from openai.types.responses.response_output_text import (
    ResponseOutputText,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from pydantic import ValidationError

from chatkit.agents import (
    AgentContext,
    accumulate_text,
    stream_agent_response,
)
from chatkit.events import (
    AssistantMessageContentPartAnnotationAdded,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
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
    HiddenContextItem,
    Page,
    SDKHiddenContextItem,
    SearchTask,
    ThoughtTask,
    URLSource,
    WidgetItem,
    Workflow,
    WorkflowItem,
)
from chatkit.turns import PersistWorkflow
from chatkit.widgets import Card, Text
from tests.helpers.agent_stream import RunResult, all_events, make_thread_and_store
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.server import DEFAULT_CONTEXT

thread, mock_store = make_thread_and_store()


def test_agent_context_requires_an_explicit_resume_choice():
    with pytest.raises(ValidationError, match="resume_from"):
        AgentContext.model_validate({
            "thread": thread,
            "store": mock_store,
            "request_context": None,
        })


@pytest.mark.parametrize("field", ["resum_from", "workflow_item"])
def test_agent_context_identifies_unknown_constructor_field(field: str):
    extra: dict[str, Any] = {field: None}
    with pytest.raises(ValidationError) as error:
        AgentContext(
            thread=thread,
            store=mock_store,
            request_context=None,
            resume_from=None,
            **extra,
        )
    (issue,) = error.value.errors()
    assert (issue["type"], issue["loc"]) == ("extra_forbidden", (field,))


def test_agent_context_accepts_declared_app_fields():
    class AppContext(AgentContext[None]):
        locale: str

    context = AppContext(
        thread=thread,
        store=mock_store,
        request_context=None,
        resume_from=None,
        locale="es",
    )
    assert context.locale == "es"


def test_agent_context_rejects_workflow_from_another_thread():
    other_workflow = WorkflowItem(
        id="other_workflow",
        thread_id="other_thread",
        created_at=datetime.now(UTC),
        workflow=Workflow(type="custom", tasks=[]),
    )
    with pytest.raises(ValidationError, match="thread"):
        AgentContext(
            thread=thread,
            store=mock_store,
            request_context=None,
            resume_from=other_workflow,
        )


async def test_first_task_updates_an_empty_saved_workflow():
    previous = WorkflowItem(
        id="saved_workflow",
        thread_id=thread.id,
        created_at=datetime.now(UTC),
        workflow=Workflow(type="custom", tasks=[]),
    )
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=previous
    )
    task = CustomTask(title="First step")

    await context.add_workflow_task(task)

    assert context._events.get_nowait() == ThreadItemUpdatedEvent(
        item_id=previous.id,
        update=WorkflowTaskAdded(task_index=0, task=task),
    )
    assert previous.workflow.tasks == []


@pytest.mark.parametrize("second_title", ["Processing", "Next step"])
async def test_added_workflow_task_uses_appended_index(second_title):
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()
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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()
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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()
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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()
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
        resume_from=None,
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

    context.request_client_tool(
        tool_context(1), name="get_selection", arguments={}, continuation="local"
    )
    with pytest.raises(ValueError, match="Only one client tool"):
        context.request_client_tool(
            tool_context(2), name="get_selection", arguments={}, continuation="local"
        )

    # The SDK can turn a callback exception into a tool result and complete.
    result = RunResult()
    result.done()
    with pytest.raises(ValueError, match="Only one client tool"):
        await all_events(stream_agent_response(context, result))


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("continued", "must stop"),
        ("changed_arguments", "preserve the original"),
        ("duplicate_output", "one output"),
        ("chat_completions", "real OpenAI Responses response ID"),
    ],
)
async def test_client_tool_handoff_requires_native_continuation(
    defect: str, message: str, monkeypatch: pytest.MonkeyPatch
):
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    call = ResponseFunctionToolCall(
        id="fc_browser",
        type="function_call",
        call_id="call_browser",
        name="get_selection",
        arguments="{}",
    )
    tool_context = ToolContext(
        context=context,
        tool_name=call.name,
        tool_arguments=call.arguments,
        tool_call_id=call.call_id,
        tool_call=call,
    )
    context.request_client_tool(
        tool_context,
        name="read_canvas_selection",
        arguments={"kind": "selection"},
        continuation="responses" if defect == "chat_completions" else "local",
    )
    output = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": "None",
    }
    response_id = "__fake_id__" if defect == "chat_completions" else "resp_browser"
    result = RunResult()
    result.interruptions = []
    result.raw_responses = [
        ModelResponse(
            output=[] if defect == "continued" else [call],
            usage=Usage(),
            response_id=response_id,
        )
    ]
    result.new_items = [
        ToolCallOutputItem(agent=Agent(name="example"), raw_item=output, output=None)
    ]
    replay_call = call.model_dump(mode="json", exclude_none=True)
    if defect == "changed_arguments":
        replay_call["arguments"] = '{"kind": "selection"}'
    items = [replay_call, output]
    if defect == "duplicate_output":
        items.append(output)
    monkeypatch.setattr(result, "to_input_list", Mock(return_value=items))
    result.done()

    with pytest.raises(ValueError, match=message):
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

    result = RunResult()
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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

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

    assert isinstance(event, ThreadItemAddedEvent)

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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

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

    assert {getattr(event1, "type", None), getattr(event2, "type", None)} == {
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


@pytest.mark.parametrize("throw_guardrail", ["input", "output"])
async def test_stream_agent_response_yields_item_removed_event(throw_guardrail):
    previous = WorkflowItem(
        id="saved_workflow",
        thread_id=thread.id,
        created_at=datetime.now(UTC),
        workflow=Workflow(type="custom", tasks=[CustomTask(title="Previous step")]),
    )
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=previous
    )
    result = RunResult()
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

    events = []
    seen = set()
    async with asyncio.timeout(2):
        async for event in iterator:
            events.append(event)
            if isinstance(event, (ThreadItemAddedEvent, ThreadItemDoneEvent)):
                seen.add(event.item.id)
            if seen == {"1", "2", "3"}:
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
    assert any(
        isinstance(event, ThreadItemReplacedEvent) and event.item.id == previous.id
        for event in events
    )


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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()
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


async def test_active_web_search_at_end_of_stream_finishes_workflow():
    store = Mock(
        generate_item_id=Mock(return_value="search_workflow"),
        load_thread_items=AsyncMock(return_value=Page()),
        add_thread_item=AsyncMock(),
    )
    context = AgentContext(
        thread=thread, store=store, request_context=None, resume_from=None
    )
    result = RunResult()
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
        aclosing(stream_agent_response(context, result)) as events,
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
    store.load_thread_items.assert_not_awaited()
    store.add_thread_item.assert_not_awaited()


async def test_web_search_finishes_resumed_workflow_with_replacement(tmp_path: Path):
    thought = ThoughtTask(content="Existing reasoning")
    previous = WorkflowItem(
        id="saved_workflow",
        thread_id=thread.id,
        created_at=datetime.now(UTC),
        workflow=Workflow(type="reasoning", tasks=[thought]),
    )
    path = tmp_path / "thread.sqlite"
    store = SQLiteStore(path)
    await store.save_thread(thread, DEFAULT_CONTEXT)
    await store.add_thread_item(thread.id, previous, DEFAULT_CONTEXT)
    unrelated = previous.model_copy(
        update={
            "id": "unrelated_workflow",
            "created_at": previous.created_at + timedelta(seconds=1),
        }
    )
    await store.add_thread_item(thread.id, unrelated, DEFAULT_CONTEXT)

    reopened_store = SQLiteStore(path)
    saved = await reopened_store.load_item(thread.id, previous.id, DEFAULT_CONTEXT)
    assert isinstance(saved, WorkflowItem)
    context = AgentContext(
        thread=thread,
        store=reopened_store,
        request_context=DEFAULT_CONTEXT,
        resume_from=saved,
    )
    result = RunResult()
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
    await reopened_store.save_item(thread.id, replaced.item, DEFAULT_CONTEXT)
    assert (
        await reopened_store.load_item(thread.id, previous.id, DEFAULT_CONTEXT)
        == replaced.item
    )
    assert (
        await reopened_store.load_item(thread.id, unrelated.id, DEFAULT_CONTEXT)
        == unrelated
    )


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
    context = AgentContext(
        thread=thread, store=store, request_context=None, resume_from=previous
    )
    active = context.workflow_item
    assert active == previous
    assert active is not previous
    result = RunResult()

    async with aclosing(stream_agent_response(context, result)) as stream:
        if finish == "explicit":
            # An agent tool can update and finish progress before the relay is consumed.
            next_task = CustomTask(title="Next step")
            await context.add_workflow_task(next_task)
            await context.end_workflow(expanded=True)
            update = await anext(stream)
            assert isinstance(update, ThreadItemUpdatedEvent)
            assert update.item_id == previous.id
            assert update.update == WorkflowTaskAdded(task_index=1, task=next_task)
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

    store.load_thread_items.assert_not_awaited()
    store.add_thread_item.assert_not_awaited()
    assert previous.workflow.tasks == [CustomTask(title="Previous step")]
    assert previous.workflow.summary is None
    if finish == "remain_open":
        assert events == [PersistWorkflow(item=previous, replace=True)]
    else:
        assert isinstance(events[0], ThreadItemReplacedEvent)
        item = events[0].item
        assert isinstance(item, WorkflowItem)
        assert item.id == previous.id
        assert isinstance(item.workflow.summary, DurationSummary)
        assert item.workflow.expanded is (finish == "explicit")
        assert context.workflow_item is None
    store.save_item.assert_not_awaited()


@pytest.mark.parametrize("item_type", [HiddenContextItem, SDKHiddenContextItem])
async def test_server_only_context_leaves_the_display_workflow_open(item_type):
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    await context.start_workflow(
        Workflow(type="custom", tasks=[CustomTask(title="Checking availability")])
    )
    workflow = context.workflow_item
    assert workflow is not None
    hidden = ThreadItemDoneEvent(
        item=item_type(
            id="private",
            thread_id=thread.id,
            created_at=datetime.now(UTC),
            content="note",
        )
    )
    await context.stream(hidden)
    result = RunResult()
    result.done()

    events = await all_events(stream_agent_response(context, result))
    assert events == [
        ThreadItemAddedEvent(item=workflow),
        hidden,
        PersistWorkflow(item=workflow, replace=False),
    ]
    assert context.workflow_item is workflow


async def test_prequeued_workflow_completion_does_not_reopen_workflow():
    store = Mock(
        generate_item_id=Mock(return_value="queued_workflow"),
        load_thread_items=AsyncMock(return_value=Page()),
        add_thread_item=AsyncMock(),
    )
    context = AgentContext(
        thread=thread, store=store, request_context=None, resume_from=None
    )
    await context.start_workflow(
        Workflow(type="custom", tasks=[CustomTask(title="Queued step")])
    )
    await context.end_workflow()
    result = RunResult()
    result.done()

    added, done = await all_events(stream_agent_response(context, result))
    assert isinstance(added, ThreadItemAddedEvent)
    assert isinstance(done, ThreadItemDoneEvent)
    assert added.item.id == done.item.id
    assert context.workflow_item is None
    store.add_thread_item.assert_not_awaited()
    with pytest.raises(asyncio.QueueShutDown):
        await context.stream(done)


async def test_fresh_stream_ignores_saved_workflow():
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
    context = AgentContext(
        thread=thread, store=store, request_context=None, resume_from=None
    )
    await context.start_workflow(
        Workflow(type="custom", tasks=[CustomTask(title="Queued step")])
    )
    result = RunResult()
    result.done()

    added, persisted = await all_events(stream_agent_response(context, result))
    assert isinstance(added, ThreadItemAddedEvent)
    assert isinstance(added.item, WorkflowItem)
    assert added.item.id == "queued_workflow"
    assert context.workflow_item is added.item
    store.load_thread_items.assert_not_awaited()
    assert persisted == PersistWorkflow(item=added.item, replace=False)
    store.add_thread_item.assert_not_awaited()
    store.save_item.assert_not_awaited()


async def test_cancelled_web_search_relay_closes_the_pending_model_read():
    store = Mock()
    store.generate_item_id.return_value = "workflow"
    store.load_thread_items = AsyncMock(return_value=Page())
    store.add_thread_item = AsyncMock()
    context = AgentContext(
        thread=thread, store=store, request_context=None, resume_from=None
    )
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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

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
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

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
    previous = WorkflowItem(
        id="wf_1",
        created_at=datetime.now(),
        workflow=Workflow(type="custom", tasks=[], summary=CustomSummary(title="Test")),
        thread_id=thread.id,
    )
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=previous
    )
    result = RunResult()

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

    assert isinstance(event, ThreadItemReplacedEvent)
    assert context.workflow_item is None
    assert event.item.type == "workflow"
    assert event.item.id == previous.id
    assert event.item.workflow.summary == CustomSummary(title="Test")

    try:
        while True:
            await anext(stream)
    except StopAsyncIteration:
        pass

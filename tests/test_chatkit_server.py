import asyncio
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any, AsyncIterator, cast

import pytest

from chatkit.errors import ErrorCode
from chatkit.events import (
    AssistantMessageContentPartTextDelta,
    ProgressUpdateEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    WorkflowTaskAdded,
)
from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    CustomTask,
    FeedbackKind,
    InferenceOptions,
    ThreadMetadata,
    UserMessageInput,
    UserMessageTextContent,
    WidgetItem,
    Workflow,
    WorkflowItem,
)
from chatkit.requests import (
    ItemFeedbackParams,
    ItemsFeedbackReq,
    ThreadCreateParams,
    ThreadsCreateReq,
)
from chatkit.server import StreamingResult
from chatkit.store import (
    NotFoundError,
    StoreItemType,
    default_generate_id,
)
from chatkit.turns import (
    Turn,
)
from chatkit.widget_stream import stream_widget
from chatkit.widgets import Card, Text
from tests._types import RequestContext
from tests.helpers.server import (
    DEFAULT_CONTEXT,
    make_server,
)


async def test_stream_cancellation_persists_pending_assistant_message_and_hidden_context():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemAddedEvent(
            item=AssistantMessageItem(
                id="assistant-message-pending",
                created_at=datetime.now(UTC),
                content=[AssistantMessageContent(text="Hello, ")],
                thread_id=thread.id,
            )
        )
        yield ThreadItemUpdatedEvent(
            item_id="assistant-message-pending",
            update=AssistantMessageContentPartTextDelta(
                content_index=0,
                delta="World!",
            ),
        )
        raise asyncio.CancelledError()

    with make_server(responder) as server:
        # Allow hidden_context id generation in this test store
        original_generate_item_id = server.store.generate_item_id

        def generate_item_id(
            item_type: StoreItemType, thread: ThreadMetadata, context: Any
        ):
            if item_type == "sdk_hidden_context":
                return default_generate_id("sdk_hidden_context")
            return original_generate_item_id(item_type, thread, context)

        server.store.generate_item_id = generate_item_id  # type: ignore[method-assign]

        stream = await server.process(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            ),
            DEFAULT_CONTEXT,
        )
        assert isinstance(stream, StreamingResult)

        events: list[ThreadStreamEvent] = []
        with pytest.raises(asyncio.CancelledError):  # noqa: PT012
            async for raw in stream:
                events.append(raw)

        thread = next(e.thread for e in events if e.type == "thread.created")
        items = await server.store.load_thread_items(
            thread.id, None, 1, "desc", DEFAULT_CONTEXT
        )
        hidden_context_item = items.data[-1]
        assert hidden_context_item.type == "sdk_hidden_context"
        assert (
            hidden_context_item.content
            == "The user cancelled the stream. Stop responding to the prior request."
        )

        assistant_message_item = await server.store.load_item(
            thread.id, "assistant-message-pending", DEFAULT_CONTEXT
        )
        assert assistant_message_item.type == "assistant_message"
        assert assistant_message_item.content[0].text == "Hello, World!"


async def test_stream_cancellation_does_not_persist_pending_empty_assistant_message():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemAddedEvent(
            item=AssistantMessageItem(
                id="assistant-message-pending",
                created_at=datetime.now(UTC),
                content=[],
                thread_id=thread.id,
            )
        )
        raise asyncio.CancelledError()

    with make_server(responder) as server:
        original_generate_item_id = server.store.generate_item_id

        def generate_item_id(
            item_type: StoreItemType, thread: ThreadMetadata, context: Any
        ):
            if item_type == "sdk_hidden_context":
                return default_generate_id("sdk_hidden_context")
            return original_generate_item_id(item_type, thread, context)

        server.store.generate_item_id = generate_item_id  # type: ignore[method-assign]

        stream = await server.process(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            ),
            DEFAULT_CONTEXT,
        )
        assert isinstance(stream, StreamingResult)

        events: list[ThreadStreamEvent] = []
        with pytest.raises(asyncio.CancelledError):  # noqa: PT012
            async for raw in stream:
                events.append(raw)

        thread = next(e.thread for e in events if e.type == "thread.created")
        items = await server.store.load_thread_items(
            thread.id, None, 1, "desc", DEFAULT_CONTEXT
        )
        hidden_context_item = items.data[-1]
        assert hidden_context_item.type == "sdk_hidden_context"
        assert (
            hidden_context_item.content
            == "The user cancelled the stream. Stop responding to the prior request."
        )

        with pytest.raises(NotFoundError):
            await server.store.load_item(
                thread.id, "assistant-message-pending", DEFAULT_CONTEXT
            )


async def test_workflow_task_not_duplicated_on_done_event():
    """
    Regression test to make sure pending item updates do not modify the
    origin item that was streamed.
    """

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        workflow_item = WorkflowItem(
            id="workflow-item",
            created_at=datetime.now(UTC),
            thread_id=thread.id,
            workflow=Workflow(type="custom", tasks=[]),
        )

        yield ThreadItemAddedEvent(item=workflow_item)

        task = CustomTask(title="foo", content="bar")
        yield ThreadItemUpdatedEvent(
            item_id=workflow_item.id,
            update=WorkflowTaskAdded(task=task, task_index=0),
        )

        workflow_item.workflow.tasks.append(task)
        yield ThreadItemDoneEvent(item=workflow_item)

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )

        thread = next(e.thread for e in events if e.type == "thread.created")
        workflow_done_event = next(
            e
            for e in events
            if isinstance(e, ThreadItemDoneEvent) and isinstance(e.item, WorkflowItem)
        )
        workflow_done_item = cast(WorkflowItem, workflow_done_event.item)
        assert len(workflow_done_item.workflow.tasks) == 1
        assert workflow_done_item.workflow.tasks[0].title == "foo"

        stored = await server.store.load_item(
            thread.id, workflow_done_item.id, DEFAULT_CONTEXT
        )
        assert isinstance(stored, WorkflowItem)
        stored_workflow = cast(WorkflowItem, stored)
        assert len(stored_workflow.workflow.tasks) == 1
        assert stored_workflow.workflow.tasks[0].title == "foo"


async def test_flows_context_to_responder():
    responder_context = None
    add_feedback_context = None

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        nonlocal responder_context
        responder_context = turn.context
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="msg_1",
                created_at=datetime.now(UTC),
                content=[AssistantMessageContent(text="Hello, world!")],
                thread_id=thread.id,
            ),
        )

    def add_feedback(
        thread_id: str, item_ids: list[str], feedback: FeedbackKind, context: Any
    ) -> None:
        nonlocal add_feedback_context
        add_feedback_context = context

    context = RequestContext(user_id="text_ctx")
    with make_server(responder, add_feedback) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello, world!")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            ),
            context,
        )
        assert responder_context == context
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        item = next(
            event.item
            for event in events
            if event.type == "thread.item.done"
            and event.item.type == "assistant_message"
        )
        await server.process_non_streaming(
            ItemsFeedbackReq(
                params=ItemFeedbackParams(
                    thread_id=thread.id,
                    item_ids=[item.id],
                    kind="positive",
                )
            ),
            context,
        )
        assert add_feedback_context == context


async def test_respond_with_tool_status():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ProgressUpdateEvent(text="Tool status")
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="msg_4",
                content=[AssistantMessageContent(text="Hello, world!")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello, world!")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )

        assert len(events) == 5
        assert events[0].type == "thread.created"
        assert events[1].type == "thread.item.done"
        assert events[2].type == "stream_options"
        assert events[3].type == "progress_update"
        assert events[4].type == "thread.item.done"


async def test_empty_widget_producer_allows_turn_to_continue():
    async def empty_widget():
        return
        yield Card(children=[])

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        async with aclosing(stream_widget(turn.thread, empty_widget())) as events:
            async for event in events:
                yield event
        yield ProgressUpdateEvent(text="Still responding")

    with make_server(respond) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )

    assert [event.type for event in events] == [
        "thread.created",
        "thread.item.done",
        "stream_options",
        "progress_update",
    ]
    assert events[-1] == ProgressUpdateEvent(text="Still responding")


async def test_returns_widget_item_generator():
    thread = ThreadMetadata(
        id="test-thread-id",
        title="Test thread",
        metadata={},
        created_at=datetime.now(UTC),
    )

    def render_widget(i: int) -> Card:
        return Card(children=[Text(id="text", value="Hello, world"[:i])])

    async def widget_generator():
        yield render_widget(0)
        yield render_widget(3)
        yield render_widget(6)
        yield render_widget(12)

    events = [event async for event in stream_widget(thread, widget_generator())]

    assert len(events) == 5
    assert isinstance(events[0], ThreadItemAddedEvent)
    assert isinstance(events[0].item, WidgetItem)
    assert events[0].item.widget == Card(children=[Text(id="text", value="")]).build()

    assert isinstance(events[1], ThreadItemUpdatedEvent)
    assert events[1].update.type == "widget.streaming_text.value_delta"
    assert events[1].update.component_id == "text"
    assert events[1].update.delta == "Hel"

    assert isinstance(events[2], ThreadItemUpdatedEvent)
    assert events[2].update.type == "widget.streaming_text.value_delta"
    assert events[2].update.component_id == "text"
    assert events[2].update.delta == "lo,"

    assert isinstance(events[3], ThreadItemUpdatedEvent)
    assert events[3].update.type == "widget.streaming_text.value_delta"
    assert events[3].update.component_id == "text"
    assert events[3].update.delta == " world"

    assert isinstance(events[4], ThreadItemDoneEvent)
    assert isinstance(events[4].item, WidgetItem)
    assert (
        events[4].item.widget
        == Card(children=[Text(id="text", value="Hello, world")]).build()
    )


async def test_returns_widget_item_generator_full_replace():
    thread = ThreadMetadata(
        id="test-thread-id",
        title="Test thread",
        metadata={},
        created_at=datetime.now(UTC),
    )

    async def widget_generator():
        yield Card(children=[Text(id="text", value="Hello")])
        yield Card(children=[Text(id="text", value="World")])

    events = [event async for event in stream_widget(thread, widget_generator())]

    assert len(events) == 3
    assert isinstance(events[0], ThreadItemAddedEvent)
    assert isinstance(events[0].item, WidgetItem)
    assert (
        events[0].item.widget == Card(children=[Text(id="text", value="Hello")]).build()
    )

    assert isinstance(events[1], ThreadItemUpdatedEvent)
    assert events[1].update.type == "widget.root.updated"
    assert (
        events[1].update.widget
        == Card(children=[Text(id="text", value="World")]).build()
    )

    assert isinstance(events[2], ThreadItemDoneEvent)
    assert isinstance(events[2].item, WidgetItem)
    assert (
        events[2].item.widget == Card(children=[Text(id="text", value="World")]).build()
    )


async def test_raising_in_responder_yields_error_event():
    async def responder(
        _turn: Turn[RequestContext],
    ) -> AsyncIterator[ThreadStreamEvent]:
        raise ValueError("Test error")
        yield

    with make_server(responder) as server:
        result = await server.process(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Test error")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            ),
            RequestContext(user_id="test_user"),
        )
        assert isinstance(result, StreamingResult)
        iter = result.__aiter__()

        # Yield an errro
        async for e in iter:
            if e.type == "error":
                assert e.code == ErrorCode.STREAM_ERROR
                break
        else:
            assert False, "Expected error event"

        # Then finish
        try:
            await iter.__anext__()
        except StopAsyncIteration:
            pass
        else:
            assert False, "Expected StopAsyncIteration"

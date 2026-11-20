import json
from datetime import UTC, datetime
from typing import Any, AsyncIterator, cast

import pytest

from chatkit.events import (
    ProgressUpdateEvent,
    ThreadCreatedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemReplacedEvent,
    ThreadStreamEvent,
    ThreadUpdatedEvent,
)
from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    InferenceOptions,
    LockedStatus,
    Page,
    Thread,
    ThreadMetadata,
    ToolChoice,
    UserMessageInput,
    UserMessageItem,
    UserMessageTextContent,
    WidgetItem,
)
from chatkit.protocol import encode_message
from chatkit.requests import (
    ItemsListParams,
    ItemsListReq,
    ThreadAddUserMessageParams,
    ThreadCreateParams,
    ThreadDeleteParams,
    ThreadGetByIdParams,
    ThreadListParams,
    ThreadRetryAfterItemParams,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsDeleteReq,
    ThreadsGetByIdReq,
    ThreadsListReq,
    ThreadsRetryAfterItemReq,
    ThreadsUpdateReq,
    ThreadUpdateParams,
)
from chatkit.server import StreamingResult
from chatkit.store import (
    ConflictError,
    NotFoundError,
)
from chatkit.turns import (
    RetryTurn,
    Turn,
    UserMessageTurn,
)
from chatkit.widgets import Card, Text
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.server import (
    DEFAULT_CONTEXT,
    make_server,
)


async def test_creates_thread():
    async def responder(
        _turn: Turn[RequestContext],
    ) -> AsyncIterator[ThreadStreamEvent]:
        return
        yield

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello, world!")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                        quoted_text="Quoted text",
                    )
                )
            )
        )

        thread_created_event = next(
            event for event in events if event.type == "thread.created"
        )
        assert thread_created_event.thread.title is None

        item_done_event = next(
            event.item for event in events if event.type == "thread.item.done"
        )

        assert item_done_event.type == "user_message"
        assert item_done_event.content[0].text == "Hello, world!"
        assert item_done_event.quoted_text == "Quoted text"


async def test_saves_thread_title():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield await server.update_thread(
            turn.thread.id, {"title": "Updated title"}, turn.context
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
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        assert (
            await server.store.load_thread(
                thread.id, RequestContext(user_id="test_user")
            )
        ).title == "Updated title"
        assert events[-1].type == "thread.updated"
        assert events[-1].thread.title == "Updated title"


async def test_saves_thread_metadata():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield await server.update_thread(
            turn.thread.id, {"metadata_set": {"test_key": "test_value"}}, turn.context
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
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        assert (await server.store.load_thread(thread.id, DEFAULT_CONTEXT)).metadata[
            "test_key"
        ] == "test_value"
        assert events[-1].type == "thread.updated"


async def test_saves_thread_locked_fields():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield await server.update_thread(
            turn.thread.id, {"status": LockedStatus(reason="Because")}, turn.context
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
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        loaded = await server.store.load_thread(
            thread.id, RequestContext(user_id="test_user")
        )
        assert loaded.status == LockedStatus(reason="Because")
        assert events[-1].type == "thread.updated"
        assert events[-1].thread.status == LockedStatus(reason="Because")


async def test_saves_allowed_image_domains_and_streams_thread_updated():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield await server.update_thread(
            turn.thread.id,
            {"allowed_image_domains": ["example.com", "images.example.com"]},
            turn.context,
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
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        loaded = await server.store.load_thread(thread.id, DEFAULT_CONTEXT)
        assert loaded.allowed_image_domains == ["example.com", "images.example.com"]
        assert events[-1].type == "thread.updated"
        assert events[-1].thread.allowed_image_domains == [
            "example.com",
            "images.example.com",
        ]


async def test_omits_unset_allowed_image_domains_in_created_and_updated_json_events():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield await server.update_thread(
            turn.thread.id, {"title": "Updated title"}, turn.context
        )

    with make_server(responder) as server:
        stream = await server.process(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello, world!")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            ),
            DEFAULT_CONTEXT,
        )
        assert isinstance(stream, StreamingResult)

        thread_created_event: dict[str, Any] | None = None
        thread_updated_event: dict[str, Any] | None = None
        async for raw in stream:
            event = json.loads(encode_message(raw))
            if event["type"] == "thread.created":
                thread_created_event = event
            if event["type"] == "thread.updated":
                thread_updated_event = event

        assert thread_created_event is not None
        assert "allowed_image_domains" not in thread_created_event["thread"]

        assert thread_updated_event is not None
        assert thread_updated_event["thread"]["title"] == "Updated title"
        assert "allowed_image_domains" not in thread_updated_event["thread"]


async def test_emits_thread_updated_mid_stream_and_persists():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="assistant_a",
                content=[AssistantMessageContent(text="A")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

        # Second assistant message, after which thread.updated should be emitted
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="assistant_b",
                content=[AssistantMessageContent(text="B")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

        yield await server.update_thread(
            thread.id, {"title": "Mid-stream title"}, turn.context
        )

        # Continue streaming after the update event
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="assistant_c",
                content=[AssistantMessageContent(text="C")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

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

        # Find the created thread
        thread = next(e.thread for e in events if isinstance(e, ThreadCreatedEvent))

        # Ensure a thread.updated event occurs right after assistant_b
        idx_b = next(
            i
            for i, e in enumerate(events)
            if isinstance(e, ThreadItemDoneEvent)
            and isinstance(e.item, AssistantMessageItem)
            and e.item.id == "assistant_b"
        )
        assert isinstance(events[idx_b + 1], ThreadUpdatedEvent)
        updated_event = cast(ThreadUpdatedEvent, events[idx_b + 1])
        assert updated_event.thread.title == "Mid-stream title"

        # Streaming continues after the update event
        assert isinstance(events[idx_b + 2], ThreadItemDoneEvent)
        done_event = cast(ThreadItemDoneEvent, events[idx_b + 2])
        assert isinstance(done_event.item, AssistantMessageItem)
        assert done_event.item.id == "assistant_c"

        # Persisted in the store
        saved = await server.store.load_thread(thread.id, DEFAULT_CONTEXT)
        assert saved.title == "Mid-stream title"


async def test_list_threads_response():
    with make_server() as server:
        for i in range(25):
            await server.process_streaming(
                ThreadsCreateReq(
                    params=ThreadCreateParams(
                        input=UserMessageInput(
                            content=[UserMessageTextContent(text=f"Thread {i}")],
                            attachments=[],
                            inference_options=InferenceOptions(),
                        )
                    )
                )
            )
        threads = await server.process_non_streaming(
            ThreadsListReq(params=ThreadListParams(limit=20))
        )
        assert isinstance(threads, Page)
        assert len(threads.data) == 20

        # Newest first
        assert threads.data[0].created_at > threads.data[1].created_at
        assert threads.has_more is True
        assert threads.after is not None

        threads = await server.process_non_streaming(
            ThreadsListReq(params=ThreadListParams(after=threads.data[-1].id))
        )
        assert isinstance(threads, Page)
        assert len(threads.data) == 5
        assert threads.has_more is False
        assert not threads.after


async def test_list_items_response():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        for i in range(25):
            yield ThreadItemDoneEvent(
                item=UserMessageItem(
                    id=f"msg_{i}",
                    content=[UserMessageTextContent(text=f"Message {i}")],
                    attachments=[],
                    inference_options=InferenceOptions(),
                    thread_id=thread.id,
                    created_at=datetime.now(UTC),
                ),
            )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Thread 1")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        items = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(items, Page)
        assert len(items.data) == 20
        # Newest first
        assert items.data[0].created_at > items.data[1].created_at
        assert items.has_more is True
        assert items.after is not None

        items = await server.process_non_streaming(
            ItemsListReq(
                params=ItemsListParams(thread_id=thread.id, after=items.data[-1].id)
            )
        )
        assert isinstance(items, Page)
        assert len(items.data) == 6  # +1 for user message
        assert items.has_more is False


async def test_get_thread_by_id():
    with make_server() as server:
        # Create a thread by sending a ThreadCreateRequest
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Test thread")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        # Retrieve the thread by id
        loaded_thread = await server.process_non_streaming(
            ThreadsGetByIdReq(
                params=ThreadGetByIdParams(thread_id=thread.id),
            )
        )
        assert isinstance(loaded_thread, Thread)
        assert loaded_thread.id == thread.id
        assert loaded_thread.title == thread.title
        assert loaded_thread.items.data[0].type == "user_message"
        assert loaded_thread.items.data[0].content[0].text == "Test thread"


async def test_list_threads_paginated_response():
    with make_server() as server:
        for i in range(25):
            await server.process_streaming(
                ThreadsCreateReq(
                    params=ThreadCreateParams(
                        input=UserMessageInput(
                            content=[UserMessageTextContent(text=f"Thread {i}")],
                            attachments=[],
                            inference_options=InferenceOptions(),
                        )
                    )
                )
            )
        threads = await server.process_non_streaming(
            ThreadsListReq(params=ThreadListParams(limit=20))
        )
        assert isinstance(threads, Page)
        assert len(threads.data) == 20
        assert threads.has_more is True
        assert threads.after is not None

        threads = await server.process_non_streaming(
            ThreadsListReq(params=ThreadListParams(after=threads.data[-1].id))
        )
        assert isinstance(threads, Page)
        assert len(threads.data) == 5
        assert threads.has_more is False


async def test_threads_update_request():
    with make_server() as server:
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
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        updated_thread = await server.process_non_streaming(
            ThreadsUpdateReq(
                params=ThreadUpdateParams(thread_id=thread.id, title="New Title")
            )
        )
        assert isinstance(updated_thread, Thread)
        assert updated_thread.title == "New Title"


async def test_delete_thread():
    with make_server() as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Thread to delete")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        response = await server.process_non_streaming(
            ThreadsDeleteReq(params=ThreadDeleteParams(thread_id=thread.id))
        )
        assert encode_message(response) == b"{}"

        try:
            await server.process_non_streaming(
                ThreadsGetByIdReq(
                    params=ThreadGetByIdParams(thread_id=thread.id),
                )
            )
            assert False, "Expected ValueError for deleted thread"
        except NotFoundError as e:
            assert f"Thread {thread.id} not found" in str(e)


async def test_thread_item_removed_event_removes_item():
    removed_id = "msg_to_remove"

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=UserMessageItem(
                id=removed_id,
                content=[UserMessageTextContent(text="To be removed")],
                attachments=[],
                inference_options=InferenceOptions(),
                thread_id=thread.id,
                created_at=datetime.now(UTC),
            ),
        )
        yield ThreadItemRemovedEvent(item_id=removed_id)

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Trigger removal")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        assert any(
            e.type == "thread.item.removed" and e.item_id == removed_id for e in events
        )
        # The item should not be present in the store
        list_result = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(list_result, Page)
        items = list_result.data
        assert all(i.id != removed_id for i in items)


async def test_thread_item_replaced_event_replaces_item():
    original_id = "msg_to_replace"

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=UserMessageItem(
                id=original_id,
                content=[UserMessageTextContent(text="Original content")],
                attachments=[],
                inference_options=InferenceOptions(),
                thread_id=thread.id,
                created_at=datetime.now(UTC),
            ),
        )
        # Then replace it
        replacement_item = AssistantMessageItem(
            id=original_id,
            content=[AssistantMessageContent(text="This is the replacement content")],
            created_at=datetime.now(UTC),
            thread_id=thread.id,
        )
        yield ThreadItemReplacedEvent(item=replacement_item)

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Trigger replacement")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        # Verify the replacement event was generated
        assert any(
            e.type == "thread.item.replaced" and e.item.id == original_id
            for e in events
        )

        # Verify the item was replaced in the store
        list_result = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(list_result, Page)
        items = list_result.data

        # Find the replaced item
        replaced_item = next((i for i in items if i.id == original_id), None)
        assert replaced_item is not None
        assert isinstance(replaced_item, AssistantMessageItem)
        assert isinstance(replaced_item.content[0], AssistantMessageContent)
        assert replaced_item.content[0].text == "This is the replacement content"


async def test_thread_item_replaced_event_with_widget():
    widget_id = "widget_to_replace"
    original_widget = Card(children=[Text(value="Original widget content")]).build()
    replacement_widget = Card(children=[Text(value="Replaced widget content")]).build()

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=WidgetItem(
                id=widget_id,
                widget=original_widget,
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )
        # Then replace it
        yield ThreadItemReplacedEvent(
            item=WidgetItem(
                id=widget_id,
                widget=replacement_widget,
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[
                            UserMessageTextContent(text="Trigger widget replacement")
                        ],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        # Verify the replacement event was generated
        replacement_event = next(
            (
                e
                for e in events
                if e.type == "thread.item.replaced" and e.item.id == widget_id
            ),
            None,
        )
        assert replacement_event is not None
        assert isinstance(replacement_event.item, WidgetItem)
        assert replacement_event.item.widget == replacement_widget

        # Verify the item was replaced in the store
        list_result = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(list_result, Page)
        items = list_result.data

        # Find the replaced widget item
        replaced_item = next((i for i in items if i.id == widget_id), None)
        assert replaced_item is not None
        assert isinstance(replaced_item, WidgetItem)
        assert replaced_item.widget == replacement_widget


async def test_retry_after_item():
    responder_calls = []

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        responder_calls.append(turn.input)

        if len(responder_calls) == 1:
            # First response - return an assistant message
            yield ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    id="assistant_msg_1",
                    content=[AssistantMessageContent(text="First response")],
                    created_at=datetime.now(UTC),
                    thread_id=thread.id,
                ),
            )
        else:
            # Retry response - return different content
            yield ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    id="assistant_msg_2",
                    content=[AssistantMessageContent(text="Retried response")],
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
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        items_before = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(items_before, Page)
        assert len(items_before.data) == 2
        original_reply, user_message = items_before.data
        assert isinstance(original_reply, AssistantMessageItem)
        assert original_reply.content[0].type == "output_text"
        assert original_reply.content[0].text == "First response"
        assert isinstance(user_message, UserMessageItem)
        assert user_message.content[0].text == "Hello, world!"

        # Get the user message ID for retrying after it
        user_message_id = user_message.id

        # Retry after the user message
        retry_events = await server.process_streaming(
            ThreadsRetryAfterItemReq(
                params=ThreadRetryAfterItemParams(
                    thread_id=thread.id, item_id=user_message_id
                )
            )
        )

        # Verify the retry generated new response
        assert len(retry_events) == 2
        assert retry_events[0].type == "stream_options"
        assert retry_events[1].type == "thread.item.done"
        assert retry_events[1].item.type == "assistant_message"
        assert retry_events[1].item.content[0].type == "output_text"
        assert retry_events[1].item.content[0].text == "Retried response"

        first, retried = responder_calls
        assert isinstance(first, UserMessageTurn)
        assert isinstance(retried, RetryTurn)
        assert first.message == retried.message
        assert first.message.content[0].text == "Hello, world!"

        # Verify the assistant response was replaced in storage
        items_after = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(items_after, Page)
        assert len(items_after.data) == 2  # Still user message + assistant message
        retried_reply, saved_user_message = items_after.data
        assert isinstance(retried_reply, AssistantMessageItem)
        assert retried_reply.content[0].type == "output_text"
        assert retried_reply.content[0].text == "Retried response"
        assert isinstance(saved_user_message, UserMessageItem)
        assert saved_user_message.content[0].text == "Hello, world!"


async def test_retry_after_item_invalid_item():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="assistant_msg_1",
                content=[AssistantMessageContent(text="Response")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

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
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        items = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(items, Page)
        assistant_item_id = items.data[0].id

        # Try to retry after an assistant message (should fail)
        with pytest.raises(ConflictError, match="user message"):
            await server.process_streaming(
                ThreadsRetryAfterItemReq(
                    params=ThreadRetryAfterItemParams(
                        thread_id=thread.id, item_id=assistant_item_id
                    )
                )
            )


async def test_retry_after_item_with_multiple_messages():
    responder_calls = []

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        trigger = turn.input
        assert isinstance(trigger, (UserMessageTurn, RetryTurn))
        responder_calls.append(trigger)
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id=f"assistant_msg_{len(responder_calls)}",
                content=[
                    AssistantMessageContent(
                        text=f"Response to: {trigger.message.content[0].text}"
                    )
                ],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="First message")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        await server.process_streaming(
            ThreadsAddUserMessageReq(
                params=ThreadAddUserMessageParams(
                    thread_id=thread.id,
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Second message")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    ),
                )
            )
        )

        # Verify we have 4 items: user1, assistant1, user2, assistant2
        items_before = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(items_before, Page)
        assert len(items_before.data) == 4

        # Get the second user message ID to retry after it
        second_user_message_id = items_before.data[1].id  # Second user message

        # Retry after the second user message
        retry_events = await server.process_streaming(
            ThreadsRetryAfterItemReq(
                params=ThreadRetryAfterItemParams(
                    thread_id=thread.id, item_id=second_user_message_id
                )
            )
        )

        assert len(retry_events) == 2
        assert retry_events[0].type == "stream_options"
        assert retry_events[1].type == "thread.item.done"

        # Verify retry used the second user message
        assert len(responder_calls) == 3  # Original 2 + 1 retry
        retried = responder_calls[2]
        assert isinstance(retried, RetryTurn)
        assert retried.message.content[0].text == "Second message"

        # Verify only the last assistant response was removed and regenerated
        items_after = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        assert isinstance(items_after, Page)
        assert len(items_after.data) == 4  # Still 4 items
        second_reply, second_message, first_reply, first_message = items_after.data
        # First user message and response should remain unchanged
        assert isinstance(first_message, UserMessageItem)
        assert first_message.content[0].type == "input_text"
        assert first_message.content[0].text == "First message"
        assert isinstance(first_reply, AssistantMessageItem)
        assert first_reply.content[0].type == "output_text"
        assert first_reply.content[0].text == "Response to: First message"
        # Second user message should remain, but assistant response should be new
        assert isinstance(second_message, UserMessageItem)
        assert second_message.content[0].text == "Second message"
        assert isinstance(second_reply, AssistantMessageItem)
        assert second_reply.content[0].type == "output_text"
        assert second_reply.content[0].text == "Response to: Second message"
        assert second_reply.id != items_before.data[0].id


async def test_threads_create_passes_tools_to_responder():
    tool_choice = ToolChoice(id="web_search")

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        assert isinstance(turn.input, UserMessageTurn)
        message = turn.input.message
        assert message.inference_options.tool_choice is not None
        assert message.inference_options.tool_choice.id == tool_choice.id

        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="assistant_msg_tools_create",
                content=[AssistantMessageContent(text="ok")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello with tools")],
                        attachments=[],
                        inference_options=InferenceOptions(tool_choice=tool_choice),
                    )
                )
            )
        )
        # Sanity check that the request flowed through.
        assert any(e.type == "thread.item.done" for e in events)


async def test_retry_after_item_passes_tools_to_responder():
    tool_choice = ToolChoice(id="web_search")
    inputs: list[UserMessageTurn | RetryTurn] = []

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        assert isinstance(turn.input, (UserMessageTurn, RetryTurn))
        inputs.append(turn.input)
        yield ProgressUpdateEvent(text="Tool status")

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello with tools")],
                        attachments=[],
                        inference_options=InferenceOptions(tool_choice=tool_choice),
                    )
                )
            )
        )
        message = next(
            event.item
            for event in events
            if event.type == "thread.item.done" and event.item.type == "user_message"
        )
        await server.process_streaming(
            ThreadsRetryAfterItemReq(
                params=ThreadRetryAfterItemParams(
                    thread_id=message.thread_id, item_id=message.id
                )
            )
        )

    original, retried = inputs
    assert isinstance(original, UserMessageTurn)
    assert isinstance(retried, RetryTurn)
    assert original.message.inference_options.tool_choice == tool_choice
    assert retried.message.inference_options.tool_choice == tool_choice


async def test_threads_add_message_passes_tools_to_responder():
    tool_choice = ToolChoice(id="retrieval")

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        assert isinstance(turn.input, UserMessageTurn)
        message = turn.input.message
        assert message.inference_options.tool_choice is not None
        assert message.inference_options.tool_choice.id == tool_choice.id

        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="assistant_msg_tools_add",
                content=[AssistantMessageContent(text="ok")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )

    with make_server(responder) as server:
        # Create a thread first.
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Start")],
                        attachments=[],
                        inference_options=InferenceOptions(tool_choice=tool_choice),
                    )
                )
            )
        )
        thread = next(e.thread for e in events if e.type == "thread.created")

        # Add a message with tools and verify they reach the responder.
        events = await server.process_streaming(
            ThreadsAddUserMessageReq(
                params=ThreadAddUserMessageParams(
                    thread_id=thread.id,
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Second")],
                        attachments=[],
                        inference_options=InferenceOptions(tool_choice=tool_choice),
                    ),
                )
            )
        )
        # Sanity check that the request flowed through.
        assert any(e.type == "thread.item.done" for e in events)


async def test_custom_id_generators_on_server():
    class UniqueIdStore(SQLiteStore):
        def __init__(self, path):
            super().__init__(path)
            self.counter = 0

        def generate_thread_id(self, context) -> str:
            self.counter += 1
            return f"thr_custom_{self.counter}"

        def generate_item_id(
            self,
            item_type: str,
            thread: ThreadMetadata,
            context,
        ) -> str:
            self.counter += 1
            return f"{item_type}_custom_{self.counter}_{thread.id}"

    async def responder(
        _turn: Turn[RequestContext],
    ) -> AsyncIterator[ThreadStreamEvent]:
        return
        yield

    with make_server(responder) as server:
        # Swap in our custom ID-generating store, reusing the same DB path.
        db_path = cast(SQLiteStore, server.store).db_path
        server.store = UniqueIdStore(db_path)

        # Create thread and verify thread id and user message id
        result = await server.process(
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
        assert isinstance(result, StreamingResult)
        events = [event async for event in result]

        thread_created = next(e for e in events if e.type == "thread.created")
        assert thread_created.thread.id == "thr_custom_1"

        user_message = next(
            e.item
            for e in events
            if e.type == "thread.item.done" and e.item.type == "user_message"
        )
        assert user_message.id == "message_custom_2_thr_custom_1"

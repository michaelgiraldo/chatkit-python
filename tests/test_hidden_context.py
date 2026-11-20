from collections.abc import AsyncIterator
from datetime import datetime

from chatkit.turns import Turn, UserMessageTurn
from chatkit.types import (
    HiddenContextItem,
    InferenceOptions,
    Page,
    SDKHiddenContextItem,
    Thread,
    ThreadCreateParams,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemReplacedEvent,
    ThreadsCreateReq,
    ThreadStreamEvent,
    ThreadUpdatedEvent,
    UserMessageInput,
    UserMessageTextContent,
)
from tests._types import RequestContext
from tests.helpers.server import DEFAULT_CONTEXT, make_server


async def test_stream_keeps_hidden_items_server_side():
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        assert isinstance(turn.input, UserMessageTurn)
        app_item = HiddenContextItem(
            id="app-note",
            thread_id=turn.thread.id,
            created_at=datetime.now(),
            content="app only",
        )
        sdk_item = SDKHiddenContextItem(
            id="sdk-note",
            thread_id=turn.thread.id,
            created_at=datetime.now(),
            content="sdk only",
        )
        for item in (app_item, sdk_item):
            yield ThreadItemAddedEvent(item=item)
            yield ThreadItemDoneEvent(item=item)
            yield ThreadItemReplacedEvent(item=item)
        yield ThreadUpdatedEvent(
            thread=Thread(
                **turn.thread.model_dump(exclude={"items"}),
                items=Page(data=[app_item, turn.input.message, sdk_item]),
            )
        )
        yield ThreadItemRemovedEvent(item_id=sdk_item.id)

    request = ThreadsCreateReq(
        params=ThreadCreateParams(
            input=UserMessageInput(
                content=[UserMessageTextContent(text="Hello")],
                attachments=[],
                inference_options=InferenceOptions(),
            )
        )
    )
    with make_server(respond) as server:
        created, user_message, options, updated = await server.process_streaming(
            request
        )
        assert created.type == "thread.created"
        assert user_message.type == "thread.item.done"
        assert options.type == "stream_options"
        assert updated.type == "thread.updated"
        assert updated.thread.items.data == [user_message.item]
        stored = await server.store.load_thread_items(
            created.thread.id, None, 10, "asc", DEFAULT_CONTEXT
        )
        assert [item.type for item in stored.data] == [
            "user_message",
            "hidden_context_item",
        ]

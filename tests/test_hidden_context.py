from collections.abc import AsyncIterator
from datetime import datetime

import pytest

from chatkit.events import (
    ErrorEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemReplacedEvent,
    ThreadStreamEvent,
    ThreadUpdatedEvent,
)
from chatkit.models import (
    HiddenContextItem,
    InferenceOptions,
    Page,
    SDKHiddenContextItem,
    Thread,
    UserMessageInput,
    UserMessageTextContent,
)
from chatkit.protocol import encode_message
from chatkit.requests import ThreadCreateParams, ThreadsCreateReq
from chatkit.turns import Turn, UserMessageTurn
from tests._types import RequestContext
from tests.helpers.server import DEFAULT_CONTEXT, make_server


@pytest.mark.parametrize("paginated", [False, True])
async def test_stream_keeps_hidden_items_server_side(paginated: bool):
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
                items=Page(
                    data=[app_item, turn.input.message, sdk_item],
                    has_more=paginated,
                    after=sdk_item.id if paginated else None,
                ),
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
        if paginated:
            assert isinstance(updated, ErrorEvent)
            assert b"sdk-note" not in encode_message(updated)
            return
        assert updated.type == "thread.updated"
        assert updated.thread.items.data == [user_message.item]
        stored = await server.store.load_thread_items(
            created.thread.id, None, 10, "asc", DEFAULT_CONTEXT
        )
        assert [item.type for item in stored.data] == [
            "user_message",
            "hidden_context_item",
        ]

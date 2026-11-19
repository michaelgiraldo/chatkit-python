import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Literal

import pytest
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from chatkit.server import NonStreamingResult, StreamingResult
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageItem,
    InferenceOptions,
    Thread,
    ThreadAddUserMessageParams,
    ThreadCreatedEvent,
    ThreadCreateParams,
    ThreadGetByIdParams,
    ThreadItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsGetByIdReq,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageItem,
    UserMessageTextContent,
)
from docs.examples.chatkit_response import ChatKitStreamingResponse
from tests._types import RequestContext
from tests.test_chatkit_server import DEFAULT_CONTEXT, decode_event, make_server


@pytest.mark.parametrize(
    ("asgi_version", "disconnect_at"),
    [("2.3", "read"), ("2.3", "send"), ("2.4", "send")],
)
async def test_http_disconnect_persists_partial_reply_and_allows_next_turn(
    asgi_version: str,
    disconnect_at: Literal["read", "send"],
    monkeypatch: pytest.MonkeyPatch,
):
    disconnect = asyncio.Event()
    stalled = asyncio.Event()
    finalized = asyncio.Event()
    streamed: list[ThreadStreamEvent] = []

    async def respond(
        thread: ThreadMetadata, message: UserMessageItem | None, context: RequestContext
    ) -> AsyncIterator[ThreadStreamEvent]:
        assert message is not None
        if message.content == [UserMessageTextContent(text="Continue")]:
            yield ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    id="next-reply",
                    thread_id=thread.id,
                    created_at=datetime.now(UTC),
                    content=[AssistantMessageContent(text="Next reply")],
                )
            )
            return

        try:
            yield ThreadItemAddedEvent(
                item=AssistantMessageItem(
                    id="partial-reply",
                    thread_id=thread.id,
                    created_at=datetime.now(UTC),
                    content=[AssistantMessageContent(text="Partial")],
                )
            )
            yield ThreadItemUpdatedEvent(
                item_id="partial-reply",
                update=AssistantMessageContentPartTextDelta(
                    content_index=0, delta=" reply"
                ),
            )
            if disconnect_at == "read":
                disconnect.set()
            await stalled.wait()
        finally:
            await asyncio.sleep(0)
            finalized.set()

    async def receive() -> Message:
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        elif message["type"] == "http.response.body" and message["body"]:
            event = decode_event(message["body"])
            streamed.append(event)
            if disconnect_at == "send" and isinstance(event, ThreadItemUpdatedEvent):
                if asgi_version == "2.4":
                    raise BrokenPipeError("Synthetic client disconnect")
                disconnect.set()
                await stalled.wait()

    def message_input(text: str) -> UserMessageInput:
        return UserMessageInput(
            content=[UserMessageTextContent(text=text)],
            attachments=[],
            inference_options=InferenceOptions(),
        )

    with make_server(respond) as server:
        add_thread_item = server.store.add_thread_item

        async def add_thread_item_with_io(
            thread_id: str, item: ThreadItem, context: RequestContext
        ) -> None:
            # The in-memory SQLite fixture does not suspend during a write.
            await asyncio.sleep(0)
            await add_thread_item(thread_id, item, context)

        monkeypatch.setattr(server.store, "add_thread_item", add_thread_item_with_io)
        request = ThreadsCreateReq(
            params=ThreadCreateParams(input=message_input("Start"))
        )
        result = await server.process(request.model_dump_json(), DEFAULT_CONTEXT)
        assert isinstance(result, StreamingResult)

        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": asgi_version},
        }
        response = ChatKitStreamingResponse(result)
        expected_exit = (
            pytest.raises(ClientDisconnect) if asgi_version == "2.4" else nullcontext()
        )
        async with asyncio.timeout(5):
            with expected_exit:
                await response(scope, receive, send)

        finalized_at_disconnect = finalized.is_set()
        assert any(isinstance(event, ThreadItemUpdatedEvent) for event in streamed)
        thread_id = next(
            event.thread.id
            for event in streamed
            if isinstance(event, ThreadCreatedEvent)
        )
        reload = ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread_id))
        loaded = await server.process(reload.model_dump_json(), DEFAULT_CONTEXT)
        assert isinstance(loaded, NonStreamingResult)
        partial_replies = [
            (item.id, item.content[0].text)
            for item in Thread.model_validate_json(loaded.json).items.data
            if isinstance(item, AssistantMessageItem)
        ]

        followup = ThreadsAddUserMessageReq(
            params=ThreadAddUserMessageParams(
                thread_id=thread_id, input=message_input("Continue")
            )
        )
        next_turn = await server.process_streaming(followup)
        assert [
            (event.item.id, event.item.content[0].text)
            for event in next_turn
            if isinstance(event, ThreadItemDoneEvent)
            and isinstance(event.item, AssistantMessageItem)
        ] == [("next-reply", "Next reply")]
        assert (finalized_at_disconnect, partial_replies) == (
            True,
            [("partial-reply", "Partial reply")],
        )

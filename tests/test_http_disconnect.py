import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Literal

import pytest
from starlette.responses import StreamingResponse
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
from tests._types import RequestContext
from tests.test_chatkit_server import DEFAULT_CONTEXT, decode_event, make_server


@pytest.mark.parametrize("disconnect_at", ["read", "send"])
async def test_http_disconnect_persists_partial_reply_and_allows_next_turn(
    disconnect_at: Literal["read", "send"],
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
                disconnect.set()
                await stalled.wait()

    def message_input(text: str) -> UserMessageInput:
        return UserMessageInput(
            content=[UserMessageTextContent(text=text)],
            attachments=[],
            inference_options=InferenceOptions(),
        )

    with make_server(respond) as server:
        request = ThreadsCreateReq(
            params=ThreadCreateParams(input=message_input("Start"))
        )
        result = await server.process(request.model_dump_json(), DEFAULT_CONTEXT)
        assert isinstance(result, StreamingResult)

        # In ASGI 2.3, the HTTP server reports a disconnected client via receive().
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
        }
        async with asyncio.timeout(5), aclosing(result.__aiter__()) as body:
            response = StreamingResponse(body, media_type="text/event-stream")
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

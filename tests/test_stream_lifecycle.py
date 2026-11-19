import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime

import pytest

from chatkit.errors import ErrorCode
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageItem,
    ErrorEvent,
    InferenceOptions,
    ProgressUpdateEvent,
    StreamOptionsEvent,
    ThreadCreatedEvent,
    ThreadCreateParams,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    ThreadsCreateReq,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageItem,
    UserMessageTextContent,
)
from tests._types import RequestContext
from tests.test_chatkit_server import DEFAULT_CONTEXT, decode_event, make_server


async def start_stream(server: ChatKitServer[RequestContext]) -> StreamingResult:
    request = ThreadsCreateReq(
        params=ThreadCreateParams(
            input=UserMessageInput(
                content=[UserMessageTextContent(text="Hello")],
                attachments=[],
                inference_options=InferenceOptions(),
            )
        )
    )
    result = await server.process(request.model_dump_json(), DEFAULT_CONTEXT)
    assert isinstance(result, StreamingResult)
    return result


async def read_turn_prelude(stream: AsyncIterator[bytes]) -> ThreadCreatedEvent:
    created = decode_event(await anext(stream))
    assert isinstance(created, ThreadCreatedEvent)
    user_message = decode_event(await anext(stream))
    assert isinstance(user_message, ThreadItemDoneEvent)
    assert isinstance(user_message.item, UserMessageItem)
    assert isinstance(decode_event(await anext(stream)), StreamOptionsEvent)
    return created


def partial_message(thread: ThreadMetadata) -> ThreadItemAddedEvent:
    return ThreadItemAddedEvent(
        item=AssistantMessageItem(
            id="partial-answer",
            created_at=datetime.now(),
            content=[AssistantMessageContent(text="Partial")],
            thread_id=thread.id,
        )
    )


async def assert_cancelled_turn(
    server: ChatKitServer[RequestContext], thread_id: str, expected_text: str
) -> None:
    items = await server.store.load_thread_items(
        thread_id, None, 10, "desc", DEFAULT_CONTEXT
    )
    assert [item.type for item in items.data] == [
        "sdk_hidden_context",
        "assistant_message",
        "user_message",
    ]
    assistant_message = items.data[1]
    assert isinstance(assistant_message, AssistantMessageItem)
    assert assistant_message.id == "partial-answer"
    assert assistant_message.content[0].text == expected_text


@pytest.mark.parametrize("iterator_kind", ["public", "json_events"])
async def test_stream_close_finalizes_responder_and_persists_partial_message(
    iterator_kind,
):
    finalized = asyncio.Event()

    async def responder(
        thread: ThreadMetadata, message: UserMessageItem | None, context: RequestContext
    ) -> AsyncIterator[ThreadStreamEvent]:
        try:
            yield partial_message(thread)
            yield ThreadItemUpdatedEvent(
                item_id="partial-answer",
                update=AssistantMessageContentPartTextDelta(
                    content_index=0, delta=" answer"
                ),
            )
        finally:
            finalized.set()

    with make_server(responder) as server:
        result = await start_stream(server)
        stream = result.__aiter__() if iterator_kind == "public" else result.json_events
        async with aclosing(stream):
            created = await read_turn_prelude(stream)
            assert isinstance(decode_event(await anext(stream)), ThreadItemAddedEvent)
            assert isinstance(decode_event(await anext(stream)), ThreadItemUpdatedEvent)

        assert finalized.is_set()
        await assert_cancelled_turn(server, created.thread.id, "Partial answer")

        with pytest.raises(StopAsyncIteration):
            await anext(stream)


async def test_cancelling_stream_read_finalizes_responder_and_persists_partial_message():
    waiting = asyncio.Event()
    finalized = asyncio.Event()

    async def responder(
        thread: ThreadMetadata, message: UserMessageItem | None, context: RequestContext
    ) -> AsyncIterator[ThreadStreamEvent]:
        try:
            yield partial_message(thread)
            waiting.set()
            await asyncio.Event().wait()
        finally:
            finalized.set()

    with make_server(responder) as server:
        result = await start_stream(server)
        async with aclosing(result.__aiter__()) as stream:
            created = await read_turn_prelude(stream)
            assert isinstance(decode_event(await anext(stream)), ThreadItemAddedEvent)

            async with asyncio.TaskGroup() as tasks:
                pull = tasks.create_task(anext(stream))
                await asyncio.wait_for(waiting.wait(), 5)
                pull.cancel()

            assert pull.cancelled()
            assert finalized.is_set()
            await assert_cancelled_turn(server, created.thread.id, "Partial")


@pytest.mark.parametrize("responder_fails", [False, True])
async def test_stream_can_be_read_across_tasks(responder_fails):
    async def responder(
        thread: ThreadMetadata, message: UserMessageItem | None, context: RequestContext
    ) -> AsyncIterator[ThreadStreamEvent]:
        yield ProgressUpdateEvent(text="First")
        if responder_fails:
            raise RuntimeError("Synthetic responder failure")
        yield ProgressUpdateEvent(text="Second")

    with make_server(responder) as server:
        result = await start_stream(server)
        async with aclosing(result.__aiter__()) as stream:
            await read_turn_prelude(stream)

            first = decode_event(await asyncio.create_task(anext(stream)))
            assert first == ProgressUpdateEvent(text="First")
            second = decode_event(await asyncio.create_task(anext(stream)))
            if responder_fails:
                assert second == ErrorEvent(
                    code=ErrorCode.STREAM_ERROR, allow_retry=True
                )
            else:
                assert second == ProgressUpdateEvent(text="Second")

            with pytest.raises(StopAsyncIteration):
                await asyncio.create_task(anext(stream))


async def test_stream_accepts_responder_without_aclose():
    class ProgressIterator(AsyncIterator[ThreadStreamEvent]):
        sent = False

        async def __anext__(self) -> ThreadStreamEvent:
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return ProgressUpdateEvent(text="Ready")

    def responder(
        thread: ThreadMetadata, message: UserMessageItem | None, context: RequestContext
    ) -> AsyncIterator[ThreadStreamEvent]:
        return ProgressIterator()

    with make_server(responder) as server:
        result = await start_stream(server)
        events = [decode_event(raw) async for raw in result]

    assert [event.type for event in events] == [
        "thread.created",
        "thread.item.done",
        "stream_options",
        "progress_update",
    ]
    assert events[-1] == ProgressUpdateEvent(text="Ready")

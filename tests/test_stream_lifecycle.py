import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime

import pytest

from chatkit.errors import ErrorCode
from chatkit.events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    ErrorEvent,
    ProgressUpdateEvent,
    StreamOptionsEvent,
    ThreadCreatedEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
)
from chatkit.models import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageItem,
    InferenceOptions,
    ThreadMetadata,
    URLSource,
    UserMessageInput,
    UserMessageItem,
    UserMessageTextContent,
)
from chatkit.requests import ThreadCreateParams, ThreadsCreateReq
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.turns import Turn
from tests._types import RequestContext
from tests.helpers.server import DEFAULT_CONTEXT, make_server


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
    result = await server.process(request, DEFAULT_CONTEXT)
    assert isinstance(result, StreamingResult)
    return result


async def read_turn_prelude(
    stream: AsyncIterator[ThreadStreamEvent],
) -> ThreadCreatedEvent:
    created = await anext(stream)
    assert isinstance(created, ThreadCreatedEvent)
    user_message = await anext(stream)
    assert isinstance(user_message, ThreadItemDoneEvent)
    assert isinstance(user_message.item, UserMessageItem)
    assert isinstance(await anext(stream), StreamOptionsEvent)
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


async def test_pending_text_delta_leaves_responder_content_untouched():
    part = AssistantMessageContent(text="")

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        item = AssistantMessageItem(
            id="reply",
            thread_id=turn.thread.id,
            created_at=datetime.now(UTC),
            content=[],
        )
        yield ThreadItemAddedEvent(item=item)
        yield ThreadItemUpdatedEvent(
            item_id=item.id,
            update=AssistantMessageContentPartAdded(content_index=0, content=part),
        )
        yield ThreadItemUpdatedEvent(
            item_id=item.id,
            update=AssistantMessageContentPartTextDelta(content_index=0, delta="Hi"),
        )
        part.text += "Hi"
        item.content.append(part)
        yield ThreadItemDoneEvent(item=item)

    with make_server(responder) as server:
        async with aclosing(await start_stream(server)) as stream:
            created = await read_turn_prelude(stream)
            events = [event async for event in stream]
        saved = await server.store.load_item(
            created.thread.id, "reply", DEFAULT_CONTEXT
        )
        assert isinstance(saved, AssistantMessageItem)
        assert saved.content == [AssistantMessageContent(text="Hi")]
        assert part.text == "Hi"
        assert events[-1] == ThreadItemDoneEvent(item=saved)


async def test_disconnect_saves_only_emitted_content_and_citation():
    part = AssistantMessageContent(text="Second part")
    citation = Annotation(source=URLSource(url="https://example.com", title="Source"))

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield partial_message(turn.thread)
        yield ThreadItemUpdatedEvent(
            item_id="partial-answer",
            update=AssistantMessageContentPartDone(content_index=1, content=part),
        )
        yield ThreadItemUpdatedEvent(
            item_id="partial-answer",
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=1, annotation_index=0, annotation=citation
            ),
        )
        part.text = "Unpublished edit"
        citation.source.title = "Unpublished title"
        yield ProgressUpdateEvent(text="Working")

    with make_server(responder) as server:
        async with aclosing(await start_stream(server)) as stream:
            created = await read_turn_prelude(stream)
            async for event in stream:
                if isinstance(event, ProgressUpdateEvent):
                    break
        saved = await server.store.load_item(
            created.thread.id, "partial-answer", DEFAULT_CONTEXT
        )
        assert isinstance(saved, AssistantMessageItem)
        assert [content.text for content in saved.content] == ["Partial", "Second part"]
        assert [a.source.title for a in saved.content[1].annotations] == ["Source"]
        assert part.annotations == []


async def test_stream_close_finalizes_responder_and_persists_partial_message():
    finalized = asyncio.Event()

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        try:
            yield partial_message(turn.thread)
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
        assert result.__aiter__() is result
        async with aclosing(result) as stream:
            created = await read_turn_prelude(stream)
            assert isinstance(await anext(stream), ThreadItemAddedEvent)
            assert isinstance(await anext(stream), ThreadItemUpdatedEvent)

        assert finalized.is_set()
        await assert_cancelled_turn(server, created.thread.id, "Partial answer")

        with pytest.raises(StopAsyncIteration):
            await anext(stream)


async def test_cancelling_stream_read_finalizes_responder_and_persists_partial_message():
    waiting = asyncio.Event()
    finalized = asyncio.Event()

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        try:
            yield partial_message(turn.thread)
            waiting.set()
            await asyncio.Event().wait()
        finally:
            finalized.set()

    with make_server(responder) as server:
        result = await start_stream(server)
        async with aclosing(result) as stream:
            created = await read_turn_prelude(stream)
            assert isinstance(await anext(stream), ThreadItemAddedEvent)

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
        _turn: Turn[RequestContext],
    ) -> AsyncIterator[ThreadStreamEvent]:
        yield ProgressUpdateEvent(text="First")
        if responder_fails:
            raise RuntimeError("Synthetic responder failure")
        yield ProgressUpdateEvent(text="Second")

    with make_server(responder) as server:
        result = await start_stream(server)
        async with aclosing(result) as stream:
            await read_turn_prelude(stream)

            first = await asyncio.create_task(anext(stream))
            assert first == ProgressUpdateEvent(text="First")
            second = await asyncio.create_task(anext(stream))
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

    def responder(_turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        return ProgressIterator()

    with make_server(responder) as server:
        result = await start_stream(server)
        events = [event async for event in result]

    assert [event.type for event in events] == [
        "thread.created",
        "thread.item.done",
        "stream_options",
        "progress_update",
    ]
    assert events[-1] == ProgressUpdateEvent(text="Ready")

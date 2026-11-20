import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest
from agents import RawResponsesStreamEvent, RunResultStreaming, StreamEvent
from openai.types.responses import ResponseTextDeltaEvent

from chatkit.agents import (
    AgentContext,
    ResponseStreamConverter,
    _merge_generators,
    stream_agent_response,
)
from chatkit.types import Page, ThreadMetadata
from docs.examples.stream_image_previews import stream_image_previews


@pytest.mark.parametrize("exit", ["close", "cancel", "error", "exhaustion"])
async def test_merge_awaits_pending_read_cleanup(
    exit: Literal["close", "cancel", "error", "exhaustion"],
):
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    original = RuntimeError("source failed")

    async def blocked() -> AsyncIterator[int]:
        try:
            entered.set()
            yield await asyncio.Future[int]()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    async def source() -> AsyncIterator[int]:
        await entered.wait()
        if exit == "error":
            raise original
        if exit == "close":
            yield 1
        if exit == "cancel":
            yield await asyncio.Future[int]()

    async with (
        asyncio.timeout(2),
        aclosing(_merge_generators(source(), blocked())) as events,
    ):
        if exit == "close":
            assert await anext(events) == 1
        elif exit == "cancel":
            read = asyncio.create_task(anext(events))
            await entered.wait()
            read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await read
        elif exit == "error":
            with pytest.raises(RuntimeError) as caught:
                await anext(events)
            assert caught.value is original
        else:
            assert [value async for value in events] == []

    assert cleaned.is_set()
    assert not any(
        task.get_name().startswith("chatkit-") for task in asyncio.all_tasks()
    )


def _agent_context() -> AgentContext:
    thread = ThreadMetadata(id="example", created_at=datetime.now())
    store = Mock()
    store.load_thread_items = AsyncMock(return_value=Page())
    return AgentContext(thread=thread, store=store, request_context=None)


@pytest.mark.parametrize("exit", ["close", "cancel"])
@pytest.mark.parametrize("source", ["adapter", "image_guide"])
async def test_interrupting_agent_response_cancels_runner_before_closing_events(
    exit: Literal["close", "cancel"],
    source: Literal["adapter", "image_guide"],
):
    entered = asyncio.Event()
    runner_cancelled = asyncio.Event()
    events_closed = asyncio.Event()
    result = Mock(spec=RunResultStreaming)
    result.is_complete = False

    def cancel_run() -> None:
        result.is_complete = True
        runner_cancelled.set()

    result.cancel.side_effect = cancel_run

    async def model_events() -> AsyncIterator[StreamEvent]:
        try:
            entered.set()
            if exit == "close":
                yield RawResponsesStreamEvent(
                    data=ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        item_id="example-message",
                        output_index=0,
                        content_index=0,
                        delta="Example",
                        logprobs=[],
                        sequence_number=0,
                    )
                )
            else:
                yield await asyncio.Future[StreamEvent]()
        finally:
            # A real SDK event stream can wait for its runner during finalization.
            await runner_cancelled.wait()
            events_closed.set()

    result.stream_events.side_effect = model_events
    context = _agent_context()
    stream = (
        stream_agent_response(context, result)
        if source == "adapter"
        else stream_image_previews(
            context, result, ResponseStreamConverter(partial_images=3)
        )
    )
    async with asyncio.timeout(2), aclosing(stream) as events:
        if exit == "close":
            assert (await anext(events)).type == "thread.item.updated"
        else:
            read = asyncio.create_task(anext(events))
            await entered.wait()
            read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await read

    result.cancel.assert_called_once_with()
    assert events_closed.is_set()
    assert not any(
        task.get_name().startswith("chatkit-") for task in asyncio.all_tasks()
    )


async def test_agent_response_preserves_model_error_if_cleanup_fails():
    original = RuntimeError("model stream failed")

    class ModelEvents(AsyncIterator[StreamEvent]):
        async def __anext__(self) -> StreamEvent:
            raise original

        async def aclose(self) -> None:
            raise ValueError("close failed")

    result = Mock(spec=RunResultStreaming)
    result.is_complete = False
    result.cancel.side_effect = ValueError("cancel failed")
    result.stream_events.return_value = ModelEvents()

    with pytest.raises(RuntimeError) as caught:
        await anext(stream_agent_response(_agent_context(), result))
    assert caught.value is original
    result.cancel.assert_called_once_with()

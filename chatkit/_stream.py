"""Shared ownership helpers for asynchronous streams."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from .logger import logger


async def _close_stream(stream: AsyncIterator[object]) -> None:
    # The responder contract also accepts custom iterators that have no aclose().
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


@asynccontextmanager
async def _closing_stream[TEvent](
    stream: AsyncIterator[TEvent],
) -> AsyncGenerator[AsyncIterator[TEvent]]:
    try:
        yield stream
    except BaseException:
        try:
            await _close_stream(stream)
        except Exception:
            logger.exception("Error while closing interrupted stream")
        raise
    else:
        await _close_stream(stream)

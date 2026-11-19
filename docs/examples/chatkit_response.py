import asyncio
from contextlib import suppress

from anyio import CancelScope
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from chatkit.server import StreamingResult


class ChatKitStreamingResponse(StreamingResponse):
    """Close the ChatKit stream when the HTTP response exits."""

    def __init__(self, result: StreamingResult) -> None:
        self._events = result.json_events
        super().__init__(self._events, media_type="text/event-stream")

    async def stream_response(self, send: Send) -> None:
        # Cancel the stream task once so its cleanup can still await store I/O.
        task = asyncio.create_task(super().stream_response(send))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            with CancelScope(shield=True), suppress(asyncio.CancelledError):
                await task
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self._events.aclose()

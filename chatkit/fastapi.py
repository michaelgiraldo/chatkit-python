"""HTTP responses for FastAPI apps using ChatKit."""

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from typing import Any

from anyio import CancelScope
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from .errors import (
    InvalidOperationError,
    InvalidRequestError,
    UnsupportedOperationError,
)
from .protocol import encode_message, encode_sse, parse_request
from .server import ChatKitServer, StreamingResult
from .store import ConflictError, NotFoundError


@asynccontextmanager
async def _closing_task(
    coroutine: Coroutine[Any, Any, None],
) -> AsyncIterator[asyncio.Task[None]]:
    task = asyncio.create_task(coroutine)
    try:
        yield task
    finally:
        # Cancel once so the producer's cleanup can still await store writes.
        task.cancel()
        with CancelScope(shield=True), suppress(asyncio.CancelledError):
            await task


class ChatKitStreamingResponse(StreamingResponse):
    """Close the ChatKit stream when the HTTP response exits."""

    def __init__(self, result: StreamingResult) -> None:
        self._result = result
        super().__init__(self._stream_frames(), media_type="text/event-stream")

    async def _stream_frames(self) -> AsyncIterator[bytes]:
        async for event in self._result:
            yield encode_sse(event)

    async def stream_response(self, send: Send) -> None:
        async with _closing_task(super().stream_response(send)) as stream:
            await asyncio.shield(stream)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            version = scope.get("asgi", {}).get("spec_version", "2.0")
            if scope["type"] != "http" or tuple(map(int, version.split("."))) < (2, 4):
                return await super().__call__(scope, receive, send)

            async def send_http(message: Message) -> None:
                try:
                    await send(message)
                except OSError as error:
                    raise ClientDisconnect() from error

            async with (
                _closing_task(self.listen_for_disconnect(receive)) as disconnect,
                _closing_task(self.stream_response(send_http)) as stream,
            ):
                finished, _ = await asyncio.wait(
                    (stream, disconnect), return_when=asyncio.FIRST_COMPLETED
                )
                for task in finished:
                    task.result()
            if self.background is not None:
                await self.background()
        finally:
            with CancelScope(shield=True):
                await self._result.aclose()


async def process_request[TContext](
    server: ChatKitServer[TContext],
    request: Request,
    context: TContext,
    *,
    streaming_response: Callable[
        [StreamingResult], Response
    ] = ChatKitStreamingResponse,
) -> Response:
    """Process a ChatKit endpoint request with context supplied by the app.

    Return 400 for malformed JSON, 404 for a missing resource, 409 for a
    thread-state conflict, and 422 for an invalid or unsupported request.
    Close the event stream on disconnect.

    An app-supplied streaming response takes ownership of its result. It must
    await `result.aclose()` when the response ends or the client disconnects.
    """
    try:
        operation = parse_request(await request.body())
        result = await server.process(operation, context)
    except InvalidRequestError as error:
        errors = error.validation_error.errors(
            include_input=False, include_context=False, include_url=False
        )
        if any(issue["type"] == "json_invalid" for issue in errors):
            return JSONResponse(
                {"detail": "Malformed JSON request body."}, status_code=400
            )
        detail = []
        for issue in errors:
            location = list(issue["loc"])
            issue_type = issue["type"]
            message = issue["msg"]
            if issue_type == "union_tag_invalid":
                message = "Unknown value for the type field."
                if not location:
                    message = "Unknown ChatKit operation."
            elif issue_type == "union_tag_not_found" and not location:
                message = "ChatKit operation is required."
            if (
                issue_type in ("union_tag_invalid", "union_tag_not_found")
                and not location
            ):
                location = ["type"]
            if location[:4] == [
                "threads.add_structured_input",
                "params",
                "input",
                "answers",
            ]:
                # The mapping keys are application-supplied question IDs.
                location = location[:4]
            detail.append({
                "location": location,
                "message": message,
                "type": issue_type,
            })
        return JSONResponse({"detail": detail}, status_code=422)
    except NotFoundError:
        return JSONResponse({"detail": "ChatKit resource not found."}, status_code=404)
    except ConflictError:
        return JSONResponse(
            {"detail": "The request conflicts with the current thread state."},
            status_code=409,
        )
    except (InvalidOperationError, UnsupportedOperationError) as error:
        return JSONResponse({"detail": str(error)}, status_code=422)

    if isinstance(result, StreamingResult):
        try:
            return streaming_response(result)
        except BaseException:
            await result.aclose()
            raise
    return Response(content=encode_message(result), media_type="application/json")

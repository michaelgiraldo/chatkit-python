"""HTTP responses for FastAPI apps using ChatKit."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

from anyio import CancelScope
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from .errors import (
    InvalidOperationError,
    InvalidRequestError,
    UnsupportedOperationError,
)
from .protocol import encode_message, encode_sse, parse_request
from .server import ChatKitServer, StreamingResult
from .store import ConflictError, NotFoundError


class ChatKitStreamingResponse(StreamingResponse):
    """Close the ChatKit stream when the HTTP response exits."""

    def __init__(self, result: StreamingResult) -> None:
        self._result = result
        super().__init__(self._stream_frames(), media_type="text/event-stream")

    async def _stream_frames(self) -> AsyncIterator[bytes]:
        async for event in self._result:
            yield encode_sse(event)

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
                await self._result.aclose()


async def process_request[TContext](
    server: ChatKitServer[TContext], request: Request, context: TContext
) -> Response:
    """Process a ChatKit endpoint request with context supplied by the app.

    Return 400 for malformed JSON, 422 for an invalid or unsupported request,
    and 404 for a missing resource. Close the event stream on disconnect.
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
        return ChatKitStreamingResponse(result)
    return Response(content=encode_message(result), media_type="application/json")

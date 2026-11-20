from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import FastAPI, Request, Response

from chatkit.server import ChatKitServer, StreamingResult
from chatkit.turns import Turn
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItemDoneEvent,
    ThreadStreamEvent,
)
from docs.examples.chatkit_response import ChatKitStreamingResponse
from docs.examples.in_memory_thread_store import Context, InMemoryThreadStore


class MyChatKitServer(ChatKitServer[Context]):
    async def respond(self, turn: Turn[Context]) -> AsyncIterator[ThreadStreamEvent]:
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                thread_id=turn.thread.id,
                id=self.store.generate_item_id("message", turn.thread, turn.context),
                created_at=datetime.now(),
                content=[AssistantMessageContent(text="Hello, world!")],
            ),
        )


app = FastAPI()
server = MyChatKitServer(store=InMemoryThreadStore())


@app.post("/chatkit")
async def chatkit(request: Request) -> Response:
    result = await server.process(await request.body(), context={})
    if isinstance(result, StreamingResult):
        return ChatKitStreamingResponse(result)
    return Response(content=result.json, media_type="application/json")

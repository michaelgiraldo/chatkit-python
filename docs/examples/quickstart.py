from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import FastAPI, Request, Response

from chatkit.fastapi import process_request
from chatkit.server import ChatKitServer
from chatkit.turns import Turn
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItemDoneEvent,
    ThreadStreamEvent,
)
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
    return await process_request(server, request, context={})

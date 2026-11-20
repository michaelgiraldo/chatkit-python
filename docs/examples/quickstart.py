import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse

from chatkit.actions import Action, ActionConfig
from chatkit.events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartTextDelta,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
)
from chatkit.fastapi import process_request
from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadMetadata,
    UserMessageTextContent,
    WidgetItem,
)
from chatkit.server import ChatKitServer
from chatkit.turns import RetryTurn, Turn, UserMessageTurn
from chatkit.widget_models import WidgetRoot
from chatkit.widgets import Button, Card, Text
from docs.examples.in_memory_thread_store import Context, InMemoryThreadStore

_DEMO_PAGE = Path(__file__).with_name("quickstart.html")
_SAVE_WIDGET = "demo.save_widget"
_SLOW_STEP_SECONDS = 1


def demo_widget(*, saved: bool = False) -> WidgetRoot:
    if saved:
        return Card(children=[Text(value="Widget state saved.")]).build()
    return Card(
        children=[
            Text(value="Try a server-side widget action."),
            Button(
                label="Save demo state",
                onClickAction=ActionConfig(type=_SAVE_WIDGET),
            ),
        ],
    ).build()


class MyChatKitServer(ChatKitServer[Context]):
    async def respond(self, turn: Turn[Context]) -> AsyncIterator[ThreadStreamEvent]:
        parts = (
            turn.input.message.content
            if isinstance(turn.input, (UserMessageTurn, RetryTurn))
            else []
        )
        prompt = (
            ""
            .join(
                part.text for part in parts if isinstance(part, UserMessageTextContent)
            )
            .strip()
            .casefold()
        )
        fields = {
            "id": self.store.generate_item_id("message", turn.thread, turn.context),
            "thread_id": turn.thread.id,
            "created_at": datetime.now(),
        }

        if prompt == "widget":
            yield ThreadItemDoneEvent(item=WidgetItem(widget=demo_widget(), **fields))
            return

        if prompt == "slow":
            reply = AssistantMessageItem(content=[], **fields)
            yield ThreadItemAddedEvent(item=reply)
            yield ThreadItemUpdatedEvent(
                item_id=reply.id,
                update=AssistantMessageContentPartAdded(
                    content_index=0, content=AssistantMessageContent(text="")
                ),
            )
            text = "The demo is streaming."
            yield ThreadItemUpdatedEvent(
                item_id=reply.id,
                update=AssistantMessageContentPartTextDelta(
                    content_index=0, delta=text
                ),
            )
            for step in range(1, 13):
                await asyncio.sleep(_SLOW_STEP_SECONDS)
                chunk = f" Step {step}."
                text += chunk
                yield ThreadItemUpdatedEvent(
                    item_id=reply.id,
                    update=AssistantMessageContentPartTextDelta(
                        content_index=0, delta=chunk
                    ),
                )
            yield ThreadItemDoneEvent(
                item=reply.model_copy(
                    update={"content": [AssistantMessageContent(text=text)]}
                )
            )
            return

        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                content=[AssistantMessageContent(text="Hello, world!")],
                **fields,
            ),
        )

    async def action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: Context,
    ) -> AsyncIterator[ThreadStreamEvent]:
        if action.type == _SAVE_WIDGET and sender is not None:
            yield ThreadItemReplacedEvent(
                item=sender.model_copy(update={"widget": demo_widget(saved=True)})
            )


app = FastAPI()
server = MyChatKitServer(store=InMemoryThreadStore())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_DEMO_PAGE, media_type="text/html")


@app.post("/chatkit")
async def chatkit(request: Request) -> Response:
    return await process_request(server, request, context={})

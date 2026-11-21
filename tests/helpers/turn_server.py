"""A real ChatKit server that records turn outcomes against SQLite."""

from collections.abc import AsyncIterator, Callable
from pathlib import Path

from chatkit.events import (
    ThreadStreamEvent,
)
from chatkit.models import (
    InferenceOptions,
    ThreadItem,
    UserMessageInput,
    UserMessageTextContent,
)
from chatkit.requests import (
    StreamingReq,
    ThreadCreateParams,
    ThreadsCreateReq,
)
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.turns import (
    ResponderEvent,
    Turn,
    TurnOutcome,
)
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.server import DEFAULT_CONTEXT


class RecordingServer(ChatKitServer[RequestContext]):
    def __init__(
        self,
        path: Path,
        responder: Callable[[Turn[RequestContext]], AsyncIterator[ResponderEvent]],
    ):
        super().__init__(SQLiteStore(str(path / "turns.sqlite")))
        self.responder = responder
        self.received: list[Turn[RequestContext]] = []
        self.finished: list[
            tuple[Turn[RequestContext], TurnOutcome, list[ThreadItem]]
        ] = []

    def respond(self, turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        self.received.append(turn)
        return self.responder(turn)

    async def on_turn_finished(
        self, turn: Turn[RequestContext], outcome: TurnOutcome
    ) -> None:
        items = await self.store.load_thread_items(
            turn.thread.id, None, 20, "asc", turn.context
        )
        self.finished.append((turn, outcome, items.data))

    async def open_stream(self, request: StreamingReq) -> StreamingResult:
        stream = await self.process(request, DEFAULT_CONTEXT)
        assert isinstance(stream, StreamingResult)
        return stream

    async def consume(self, request: StreamingReq) -> list[ThreadStreamEvent]:
        return [event async for event in await self.open_stream(request)]


def create_thread() -> ThreadsCreateReq:
    return ThreadsCreateReq(
        params=ThreadCreateParams(
            input=UserMessageInput(
                content=[UserMessageTextContent(text="Hello")],
                attachments=[],
                inference_options=InferenceOptions(),
            )
        )
    )

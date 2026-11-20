import sqlite3
from contextlib import contextmanager
from typing import Any, AsyncIterator, Callable

from pydantic import TypeAdapter

from chatkit.actions import Action
from chatkit.server import ChatKitServer, NonStreamingResult, StreamingResult
from chatkit.store import AttachmentStore
from chatkit.turns import Turn
from chatkit.types import (
    AudioInput,
    FeedbackKind,
    SyncCustomActionResponse,
    ThreadMetadata,
    ThreadStreamEvent,
    TranscriptionResult,
    WidgetItem,
)
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore

server_id = 0


DEFAULT_CONTEXT = RequestContext(user_id="test_user")


def decode_event(event: bytes) -> ThreadStreamEvent:
    return TypeAdapter(ThreadStreamEvent).validate_json(event.split(b"data: ")[1])


async def decode_streaming_result(
    streaming_result: StreamingResult,
) -> list[ThreadStreamEvent]:
    return [decode_event(event) async for event in streaming_result.json_events]


@contextmanager
def make_server(
    responder: Callable[[Turn[RequestContext]], AsyncIterator[ThreadStreamEvent]]
    | None = None,
    handle_feedback: Callable[[str, list[str], FeedbackKind, Any], None] | None = None,
    action_callback: Callable[
        [ThreadMetadata, Action[str, Any], WidgetItem | None, Any],
        AsyncIterator[ThreadStreamEvent],
    ]
    | None = None,
    sync_action_callback: Callable[
        [ThreadMetadata, Action[str, Any], WidgetItem | None, Any],
        SyncCustomActionResponse,
    ]
    | None = None,
    attachment_store: AttachmentStore[RequestContext] | None = None,
    transcribe_callback: Callable[[AudioInput, Any], TranscriptionResult] | None = None,
):
    global server_id
    db_path = f"file:{server_id}?mode=memory&cache=shared"
    server_id += 1
    # Keep the shared in-memory database from being deleted when the last connection is closed
    db = sqlite3.connect(db_path, uri=True)

    class TestChatKitServer(ChatKitServer[RequestContext]):
        def __init__(self):
            store = SQLiteStore(db_path)
            super().__init__(store, attachment_store, attachment_metadata_store=store)

        def action(
            self,
            thread: ThreadMetadata,
            action: Action[str, Any],
            sender: WidgetItem | None,
            context: Any,
        ) -> AsyncIterator[ThreadStreamEvent]:
            if action_callback is None:
                raise ValueError("action_callback not wired up")
            return action_callback(thread, action, sender, context)

        async def sync_action(
            self,
            thread: ThreadMetadata,
            action: Action[str, Any],
            sender: WidgetItem | None,
            context: Any,
        ) -> SyncCustomActionResponse:
            if sync_action_callback is None:
                raise ValueError("sync_action_callback not wired up")
            return sync_action_callback(thread, action, sender, context)

        def respond(
            self, turn: Turn[RequestContext]
        ) -> AsyncIterator[ThreadStreamEvent]:
            if responder is None:
                return self._empty_responder()
            return responder(turn)

        async def _empty_responder(self) -> AsyncIterator[ThreadStreamEvent]:
            return
            yield

        async def add_feedback(
            self,
            thread_id: str,
            item_ids: list[str],
            feedback: FeedbackKind,
            context: Any,
        ) -> None:
            if handle_feedback is None:
                return
            handle_feedback(thread_id, item_ids, feedback, context)

        async def transcribe(
            self, audio_input: AudioInput, context: Any
        ) -> TranscriptionResult:
            if transcribe_callback is None:
                return await super().transcribe(audio_input, context)
            return transcribe_callback(audio_input, context)

        async def process_streaming(
            self, request_obj, context: Any | None = None
        ) -> list[ThreadStreamEvent]:
            result = await self.process(
                request_obj.model_dump_json(), context or DEFAULT_CONTEXT
            )
            assert isinstance(result, StreamingResult)
            return await decode_streaming_result(result)

        async def process_non_streaming(self, request_obj, context: Any | None = None):
            result = await self.process(
                request_obj.model_dump_json(), context or DEFAULT_CONTEXT
            )
            assert isinstance(result, NonStreamingResult)
            return result

    try:
        yield TestChatKitServer()
    finally:
        db.close()

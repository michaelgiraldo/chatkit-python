from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from datetime import datetime
from pathlib import Path

import pytest

from chatkit.server import ChatKitServer, StreamingResult
from chatkit.turns import (
    ClientToolTurn,
    RetryTurn,
    StructuredInputTurn,
    Turn,
    TurnOutcome,
    UserMessageTurn,
)
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ClientToolCallItem,
    ErrorEvent,
    InferenceOptions,
    StreamingReq,
    StructuredInputAnswerSubmission,
    StructuredInputFreeform,
    StructuredInputItem,
    StructuredInputSubmission,
    ThreadAddClientToolOutputParams,
    ThreadAddStructuredInputParams,
    ThreadCreateParams,
    ThreadItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadRetryAfterItemParams,
    ThreadsAddClientToolOutputReq,
    ThreadsAddStructuredInputReq,
    ThreadsCreateReq,
    ThreadsRetryAfterItemReq,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageTextContent,
)
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.server import DEFAULT_CONTEXT


class RecordingServer(ChatKitServer[RequestContext]):
    def __init__(
        self,
        path: Path,
        responder: Callable[[Turn[RequestContext]], AsyncIterator[ThreadStreamEvent]],
    ):
        super().__init__(SQLiteStore(str(path / "turns.sqlite")))
        self.responder = responder
        self.received: list[Turn[RequestContext]] = []
        self.finished: list[
            tuple[Turn[RequestContext], TurnOutcome, list[ThreadItem]]
        ] = []

    def respond(self, turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
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


def assistant_message(
    turn: Turn[RequestContext], message_id: str, text: str
) -> AssistantMessageItem:
    return AssistantMessageItem(
        id=message_id,
        thread_id=turn.thread.id,
        created_at=datetime.now(),
        content=[AssistantMessageContent(text=text)],
    )


async def test_typed_turn_waits_for_client_tool_then_completes(tmp_path: Path):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        if isinstance(turn.input, UserMessageTurn):
            yield ThreadItemDoneEvent(
                item=ClientToolCallItem(
                    id="client-tool",
                    thread_id=turn.thread.id,
                    created_at=datetime.now(),
                    name="read_selection",
                    call_id="call-selection",
                    arguments={},
                )
            )
            return

        assert isinstance(turn.input, ClientToolTurn)
        assert turn.input.tool_call.status == "completed"
        assert turn.input.tool_call.output == "blue"
        assert (
            await server.store.load_item(
                turn.thread.id, turn.input.tool_call.id, turn.context
            )
            == turn.input.tool_call
        )
        yield ThreadItemDoneEvent(
            item=assistant_message(turn, "reply", "Selected blue")
        )

    server = RecordingServer(tmp_path, respond)
    await server.consume(create_thread())
    (initial,) = server.received
    assert isinstance(initial.input, UserMessageTurn)
    assert initial.context is DEFAULT_CONTEXT
    (initial_finished,) = server.finished
    assert initial_finished[:2] == (initial, TurnOutcome.WAITING)
    assert [item.id for item in initial_finished[2]] == [
        initial.input.message.id,
        "client-tool",
    ]

    await server.consume(
        ThreadsAddClientToolOutputReq(
            params=ThreadAddClientToolOutputParams(
                thread_id=initial.thread.id, result="blue"
            )
        )
    )
    first, continuation = server.received
    assert first is initial
    assert isinstance(continuation.input, ClientToolTurn)
    assert len(server.finished) == 2
    assert server.finished[-1][:2] == (continuation, TurnOutcome.COMPLETED)
    assert any(item.id == "reply" for item in server.finished[-1][2])


async def test_typed_turn_distinguishes_form_submission_and_retry(tmp_path: Path):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        if isinstance(turn.input, UserMessageTurn):
            yield ThreadItemDoneEvent(
                item=StructuredInputItem(
                    id="form",
                    thread_id=turn.thread.id,
                    created_at=datetime.now(),
                    inputs=[StructuredInputFreeform(id="color", question="Color?")],
                )
            )
        elif isinstance(turn.input, StructuredInputTurn):
            submitted = turn.input.item
            answer = submitted.inputs[0].answer
            assert submitted.status == "answered"
            assert answer is not None and answer.values == ["blue"]
            assert (
                await server.store.load_item(turn.thread.id, submitted.id, turn.context)
                == submitted
            )
            yield ThreadItemDoneEvent(item=assistant_message(turn, "answer", "Blue"))
        else:
            assert isinstance(turn.input, RetryTurn)
            yield ThreadItemDoneEvent(item=assistant_message(turn, "retry", "Retried"))

    server = RecordingServer(tmp_path, respond)
    await server.consume(create_thread())
    (initial,) = server.received
    assert isinstance(initial.input, UserMessageTurn)
    await server.consume(
        ThreadsAddStructuredInputReq(
            params=ThreadAddStructuredInputParams(
                thread_id=initial.thread.id,
                item_id="form",
                input=StructuredInputSubmission(
                    answers={"color": StructuredInputAnswerSubmission(values=["blue"])}
                ),
            )
        )
    )
    await server.consume(
        ThreadsRetryAfterItemReq(
            params=ThreadRetryAfterItemParams(
                thread_id=initial.thread.id, item_id=initial.input.message.id
            )
        )
    )

    first, submission, retry = server.received
    assert isinstance(submission.input, StructuredInputTurn)
    assert isinstance(retry.input, RetryTurn)
    assert retry.input.message == initial.input.message
    assert [(turn, outcome) for turn, outcome, _ in server.finished] == [
        (first, TurnOutcome.WAITING),
        (submission, TurnOutcome.COMPLETED),
        (retry, TurnOutcome.COMPLETED),
    ]
    assert [item.id for item in server.finished[-1][2]] == [
        initial.input.message.id,
        "retry",
    ]


async def test_stream_close_reports_cancelled_after_partial_reply_save(tmp_path: Path):
    responder_closed = False

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        nonlocal responder_closed
        try:
            yield ThreadItemAddedEvent(
                item=assistant_message(turn, "partial", "Unfinished reply")
            )
        finally:
            responder_closed = True

    server = RecordingServer(tmp_path, respond)
    stream = await server.open_stream(create_thread())
    async with aclosing(stream) as events:
        async for raw in events:
            if isinstance(raw, ThreadItemAddedEvent):
                break
        assert server.finished == []

    assert responder_closed
    ((turn, outcome, items),) = server.finished
    assert turn is server.received[0]
    assert outcome is TurnOutcome.CANCELLED
    partial = next(item for item in items if item.id == "partial")
    assert isinstance(partial, AssistantMessageItem)
    assert partial.content[0].text == "Unfinished reply"
    assert any(item.type == "sdk_hidden_context" for item in items)


async def test_close_after_user_message_reports_cancelled_before_respond(
    tmp_path: Path,
):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        pytest.fail("The responder should not start")
        yield

    server = RecordingServer(tmp_path, respond)
    stream = await server.open_stream(create_thread())
    async with aclosing(stream) as events:
        async for raw in events:
            event = raw
            if isinstance(event, ThreadItemDoneEvent):
                assert event.item.type == "user_message"
                break
        assert server.finished == []

    assert server.received == []
    ((turn, outcome, items),) = server.finished
    assert isinstance(turn.input, UserMessageTurn)
    assert outcome is TurnOutcome.CANCELLED
    assert turn.input.message in items


@pytest.mark.parametrize("raised", [False, True])
async def test_stream_failure_reports_failed_once(tmp_path: Path, raised: bool):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        if raised:
            raise RuntimeError("Synthetic responder failure")
        yield ErrorEvent(message="Synthetic responder failure")

    server = RecordingServer(tmp_path, respond)
    events = await server.consume(create_thread())
    assert isinstance(events[-1], ErrorEvent)
    ((turn, outcome, _),) = server.finished
    assert turn is server.received[0]
    assert outcome is TurnOutcome.FAILED

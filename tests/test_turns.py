import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime
from pathlib import Path

import pytest

from chatkit.errors import UnsupportedOperationError
from chatkit.events import (
    ErrorEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadStreamEvent,
)
from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    ClientToolCallItem,
    StructuredInputFreeform,
    StructuredInputItem,
)
from chatkit.requests import (
    ClientToolCallbackResult,
    StructuredInputAnswerSubmission,
    StructuredInputSubmission,
    ThreadAddClientToolOutputParams,
    ThreadAddStructuredInputParams,
    ThreadAddUserMessageParams,
    ThreadRetryAfterItemParams,
    ThreadsAddClientToolOutputReq,
    ThreadsAddStructuredInputReq,
    ThreadsAddUserMessageReq,
    ThreadsRetryAfterItemReq,
    ThreadsUpdateReq,
    ThreadUpdateParams,
)
from chatkit.server import StreamingResult
from chatkit.store import ConflictError, NotFoundError, ThreadStore
from chatkit.turns import (
    ClientToolInvocation,
    ClientToolRequest,
    ClientToolTurn,
    ResponderEvent,
    RetryTurn,
    StructuredInputTurn,
    Turn,
    TurnOutcome,
    TurnState,
    UserMessageTurn,
)
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.server import DEFAULT_CONTEXT
from tests.helpers.turn_server import RecordingServer, create_thread


def assistant_message(
    turn: Turn[RequestContext], message_id: str, text: str
) -> AssistantMessageItem:
    return AssistantMessageItem(
        id=message_id,
        thread_id=turn.thread.id,
        created_at=datetime.now(),
        content=[AssistantMessageContent(text=text)],
    )


def client_tool_request(turn: Turn[RequestContext], item_id: str) -> ClientToolRequest:
    return ClientToolRequest(
        item=ClientToolCallItem(
            id=item_id,
            thread_id=turn.thread.id,
            created_at=datetime.now(),
            name="read_selection",
            arguments={},
        ),
        call_id=f"call-{item_id}",
    )


def client_tool_callback(
    thread_id: str, item_id: str, output: object = None
) -> ThreadsAddClientToolOutputReq:
    return ThreadsAddClientToolOutputReq(
        params=ThreadAddClientToolOutputParams(
            thread_id=thread_id,
            result=ClientToolCallbackResult(item_id=item_id, output=output),
        )
    )


async def test_typed_turn_waits_for_client_tool_then_completes(tmp_path: Path):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        if isinstance(turn.input, UserMessageTurn):
            yield client_tool_request(turn, "client-tool")
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

    await server.consume(client_tool_callback(initial.thread.id, "client-tool", "blue"))
    first, continuation = server.received
    assert first is initial
    assert isinstance(continuation.input, ClientToolTurn)
    assert len(server.finished) == 2
    assert server.finished[-1][:2] == (continuation, TurnOutcome.COMPLETED)
    assert any(item.id == "reply" for item in server.finished[-1][2])


async def test_client_tool_routes_only_the_exact_active_item(tmp_path: Path):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        if isinstance(turn.input, UserMessageTurn):
            yield client_tool_request(turn, f"tool-{turn.input.message.id}")
            yield ThreadItemDoneEvent(
                item=assistant_message(turn, f"note-{turn.input.message.id}", "Note")
            )
        else:
            assert isinstance(turn.input, ClientToolTurn)

    def pending_tool(
        events: list[ThreadStreamEvent],
    ) -> ClientToolCallItem:
        (call,) = [
            event.item
            for event in events
            if isinstance(event, ThreadItemDoneEvent)
            and isinstance(event.item, ClientToolCallItem)
        ]
        return call

    server = RecordingServer(tmp_path, respond)
    first = pending_tool(await server.consume(create_thread()))
    foreign = pending_tool(await server.consume(create_thread()))
    second = pending_tool(
        await server.consume(
            ThreadsAddUserMessageReq(
                params=ThreadAddUserMessageParams(
                    thread_id=first.thread_id, input=create_thread().params.input
                )
            )
        )
    )
    assert first.name == second.name == foreign.name

    with pytest.raises(ConflictError):
        await server.consume(client_tool_callback(first.thread_id, first.id, "late"))
    with pytest.raises(NotFoundError):
        await server.consume(
            client_tool_callback(first.thread_id, foreign.id, "foreign")
        )

    await server.consume(client_tool_callback(first.thread_id, second.id, "current"))
    with pytest.raises(ConflictError):
        await server.consume(
            client_tool_callback(first.thread_id, second.id, "duplicate")
        )

    (resumed,) = [
        turn.input for turn in server.received if isinstance(turn.input, ClientToolTurn)
    ]
    assert resumed.tool_call.id == resumed.invocation.item_id == second.id
    assert resumed.invocation.call_id == f"call-{second.id}"
    assert resumed.tool_call.output == "current"


async def test_accepting_a_turn_fences_the_older_responder_before_the_new_stream_starts(
    tmp_path: Path,
):
    old_responder_started = asyncio.Event()
    release_old_responder = asyncio.Event()
    new_reply = "new-reply"

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        if turn.id == first.turn_id:
            old_responder_started.set()
            await release_old_responder.wait()
            yield client_tool_request(turn, "stale-first-tool")
            return
        yield ThreadItemDoneEvent(item=assistant_message(turn, new_reply, "New reply"))

    server = RecordingServer(tmp_path, respond)
    first = await server.open_stream(create_thread())
    assert first.thread_id is not None and first.turn_id is not None

    async def collect(stream: StreamingResult) -> list[ThreadStreamEvent]:
        return [event async for event in stream]

    async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
        old_stream = tasks.create_task(collect(first))
        await old_responder_started.wait()
        second = await server.open_stream(
            ThreadsAddUserMessageReq(
                params=ThreadAddUserMessageParams(
                    thread_id=first.thread_id, input=create_thread().params.input
                )
            )
        )
        assert second.turn_id is not None
        assert (
            await server.store.load_turn_state(
                first.thread_id, second.turn_id, DEFAULT_CONTEXT
            )
            is TurnState.ACCEPTED
        )
        release_old_responder.set()
        assert all(
            not isinstance(event, ThreadItemDoneEvent)
            or event.item.type == "user_message"
            for event in await old_stream
        )

    await collect(second)
    assert (
        await server.store.load_turn_state(
            first.thread_id, first.turn_id, DEFAULT_CONTEXT
        )
        is TurnState.SUPERSEDED
    )
    assert (
        await server.store.load_turn_state(
            first.thread_id, second.turn_id, DEFAULT_CONTEXT
        )
        is TurnState.COMPLETED
    )
    assert [outcome for _, outcome, _ in server.finished] == [
        TurnOutcome.SUPERSEDED,
        TurnOutcome.COMPLETED,
    ]
    _, _, final_items = server.finished[-1]
    assert [item.type for item in final_items] == [
        "user_message",
        "user_message",
        "assistant_message",
    ]
    assert final_items[-1].id == new_reply


async def test_callback_accepted_before_stream_can_be_abandoned_but_not_replayed(
    tmp_path: Path,
):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        assert isinstance(turn.input, UserMessageTurn)
        yield client_tool_request(turn, "example-tool")

    server = RecordingServer(tmp_path, respond)
    await server.consume(create_thread())
    (initial,) = server.received
    callback = client_tool_callback(
        initial.thread.id, "example-tool", "retained result"
    )
    stream = await server.open_stream(callback)
    assert stream.turn_id is not None
    assert (
        await server.store.load_turn_state(
            initial.thread.id, stream.turn_id, DEFAULT_CONTEXT
        )
        is TurnState.ACCEPTED
    )
    with pytest.raises(ConflictError):
        await server.open_stream(callback)

    await stream.aclose()
    await stream.aclose()
    assert (
        await server.store.load_turn_state(
            initial.thread.id, stream.turn_id, DEFAULT_CONTEXT
        )
        is TurnState.ABANDONED
    )
    assert await anext(stream, None) is None
    assert server.received == [initial]
    tool = await server.store.load_item(
        initial.thread.id, "example-tool", DEFAULT_CONTEXT
    )
    assert isinstance(tool, ClientToolCallItem)
    assert tool.status == "completed"
    assert tool.output == "retained result"
    invocation = await server.store.load_client_tool_invocation(
        initial.thread.id, tool.id, DEFAULT_CONTEXT
    )
    assert invocation.turn_id == initial.id
    assert invocation.continuation_turn_id == stream.turn_id
    with pytest.raises(ConflictError):
        await server.open_stream(callback)


async def test_pending_client_tool_can_resume_before_its_stream_finishes(
    tmp_path: Path,
):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        if isinstance(turn.input, UserMessageTurn):
            yield client_tool_request(turn, "example-tool")
            yield ThreadItemDoneEvent(
                item=assistant_message(turn, "stale", "Old reply")
            )
        else:
            assert isinstance(turn.input, ClientToolTurn)
            yield ThreadItemDoneEvent(
                item=assistant_message(turn, "resumed", "New reply")
            )

    server = RecordingServer(tmp_path, respond)
    async with aclosing(await server.open_stream(create_thread())) as first_stream:
        async for event in first_stream:
            if isinstance(event, ThreadItemDoneEvent) and isinstance(
                event.item, ClientToolCallItem
            ):
                break

        (initial,) = server.received
        assert (
            await server.store.load_turn_state(
                initial.thread.id, initial.id, DEFAULT_CONTEXT
            )
            is TurnState.RUNNING
        )
        callback = client_tool_callback(initial.thread.id, "example-tool")
        resumed_events = await server.consume(callback)
        assert [
            event.item.id
            for event in resumed_events
            if isinstance(event, ThreadItemDoneEvent)
        ] == ["resumed"]
        with pytest.raises(ConflictError):
            await server.open_stream(callback)
        assert await anext(first_stream, None) is None

    assert (
        await server.store.load_turn_state(
            initial.thread.id, initial.id, DEFAULT_CONTEXT
        )
        is TurnState.WAITING
    )
    assert [(turn.id, outcome) for turn, outcome, _ in server.finished] == [
        (server.received[1].id, TurnOutcome.COMPLETED),
        (initial.id, TurnOutcome.WAITING),
    ]
    with pytest.raises(NotFoundError):
        await server.store.load_item(initial.thread.id, "stale", DEFAULT_CONTEXT)


async def test_browser_item_alone_cannot_register_a_client_tool(tmp_path: Path):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield ThreadItemDoneEvent(item=client_tool_request(turn, "display-only").item)

    server = RecordingServer(tmp_path, respond)
    events = await server.consume(create_thread())
    assert isinstance(events[-1], ErrorEvent)
    ((turn, outcome, items),) = server.finished
    assert outcome is TurnOutcome.FAILED
    assert [item.type for item in items] == ["user_message"]
    with pytest.raises(NotFoundError):
        await server.store.load_client_tool_invocation(
            turn.thread.id, "display-only", turn.context
        )


async def test_client_tool_fails_closed_on_legacy_stores(tmp_path: Path):
    class LegacyStore(SQLiteStore):
        register_client_tool_call = ThreadStore.register_client_tool_call
        load_client_tool_status = ThreadStore.load_client_tool_status
        cancel_client_tool_call = ThreadStore.cancel_client_tool_call

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        assert isinstance(turn.input, UserMessageTurn)
        yield client_tool_request(turn, "example-tool")

    server = RecordingServer(tmp_path, respond)
    await server.consume(create_thread())
    (initial,) = server.received
    server.store = LegacyStore(str(tmp_path / "turns.sqlite"))

    with pytest.raises(UnsupportedOperationError):
        await server.consume(client_tool_callback(initial.thread.id, "example-tool"))
    assert server.received == [initial]

    events = await server.consume(create_thread())
    assert isinstance(events[-1], ErrorEvent)
    assert not any(
        isinstance(event, ThreadItemDoneEvent)
        and isinstance(event.item, ClientToolCallItem)
        for event in events
    )
    assert all(item.id != "example-tool" for item in server.finished[-1][2])


@pytest.mark.parametrize("end", ["close", "error_event", "raised_error"])
async def test_interrupted_client_tool_handoff_cancels_the_callback(
    tmp_path: Path, end: str
):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        assert isinstance(turn.input, UserMessageTurn)
        yield client_tool_request(turn, "example-tool")
        if end == "error_event":
            yield ErrorEvent(message="Synthetic failure")
        elif end == "raised_error":
            raise RuntimeError("Synthetic failure")

    server = RecordingServer(tmp_path, respond)
    if end == "close":
        async with aclosing(await server.open_stream(create_thread())) as events:
            async for event in events:
                if isinstance(event, ThreadItemDoneEvent) and isinstance(
                    event.item, ClientToolCallItem
                ):
                    break
    else:
        events = await server.consume(create_thread())
        assert isinstance(events[-1], ErrorEvent)

    ((turn, outcome, _),) = server.finished
    expected_outcome = TurnOutcome.CANCELLED if end == "close" else TurnOutcome.FAILED
    assert outcome is expected_outcome
    assert (
        await server.store.load_client_tool_status(
            turn.thread.id, "example-tool", turn.context
        )
        == "cancelled"
    )
    with pytest.raises(ConflictError):
        await server.consume(client_tool_callback(turn.thread.id, "example-tool"))


async def test_disconnect_settles_in_flight_client_tool_registration(tmp_path: Path):
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowStore(SQLiteStore):
        async def register_client_tool_call(
            self,
            thread_id: str,
            item: ClientToolCallItem,
            invocation: ClientToolInvocation,
            context: RequestContext,
        ) -> None:
            await super().register_client_tool_call(
                thread_id, item, invocation, context
            )
            started.set()
            await release.wait()

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        yield client_tool_request(turn, "example-tool")

    server = RecordingServer(tmp_path, respond)
    server.store = SlowStore(str(tmp_path / "turns.sqlite"))
    consuming = asyncio.create_task(server.consume(create_thread()))
    await asyncio.wait_for(started.wait(), 5)
    consuming.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await consuming

    ((turn, outcome, _),) = server.finished
    assert outcome is TurnOutcome.CANCELLED
    assert (
        await server.store.load_client_tool_status(
            turn.thread.id, "example-tool", turn.context
        )
        == "cancelled"
    )


async def test_typed_turn_distinguishes_form_submission_and_retry(tmp_path: Path):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
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


async def test_concurrent_form_submissions_only_resume_once(
    tmp_path: Path, monkeypatch
):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        if isinstance(turn.input, UserMessageTurn):
            yield ThreadItemDoneEvent(
                item=StructuredInputItem(
                    id="form",
                    thread_id=turn.thread.id,
                    created_at=datetime.now(),
                    inputs=[StructuredInputFreeform(id="color", question="Color?")],
                )
            )

    server = RecordingServer(tmp_path, respond)
    await server.consume(create_thread())
    (initial,) = server.received
    load_item = server.store.load_item
    both_read = asyncio.Barrier(2)

    async def load_same_snapshot(thread_id: str, item_id: str, context: RequestContext):
        item = await load_item(thread_id, item_id, context)
        await both_read.wait()
        return item

    monkeypatch.setattr(server.store, "load_item", load_same_snapshot)

    async def submit(color: str):
        return await server.consume(
            ThreadsAddStructuredInputReq(
                params=ThreadAddStructuredInputParams(
                    thread_id=initial.thread.id,
                    item_id="form",
                    input=StructuredInputSubmission(
                        answers={
                            "color": StructuredInputAnswerSubmission(values=[color])
                        }
                    ),
                )
            )
        )

    async with asyncio.timeout(5):
        results = await asyncio.gather(
            submit("red"), submit("blue"), return_exceptions=True
        )

    assert sum(isinstance(result, ConflictError) for result in results) == 1
    assert sum(isinstance(result, list) for result in results) == 1
    _, resumed = server.received
    assert isinstance(resumed.input, StructuredInputTurn)
    assert (
        await load_item(initial.thread.id, "form", DEFAULT_CONTEXT)
        == resumed.input.item
    )


async def test_turn_metadata_patch_does_not_overwrite_concurrent_title(tmp_path: Path):
    started = asyncio.Event()
    renamed = asyncio.Event()

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        assert turn.thread.title is None
        started.set()
        await renamed.wait()
        yield await server.update_thread(
            turn.thread.id,
            {"metadata_set": {"provider_response": "example"}},
            turn.context,
            turn_id=turn.id,
        )

    server = RecordingServer(tmp_path, respond)
    async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
        tasks.create_task(server.consume(create_thread()))
        await started.wait()
        (active_turn,) = server.received
        await server.process(
            ThreadsUpdateReq(
                params=ThreadUpdateParams(
                    thread_id=active_turn.thread.id, title="Renamed"
                )
            ),
            DEFAULT_CONTEXT,
        )
        renamed.set()

    thread = await server.store.load_thread(active_turn.thread.id, DEFAULT_CONTEXT)
    assert thread.title == "Renamed"
    assert thread.metadata == {"provider_response": "example"}


async def test_stream_close_reports_cancelled_after_partial_reply_save(tmp_path: Path):
    responder_closed = False

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
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
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
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
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        if raised:
            raise RuntimeError("Synthetic responder failure")
        yield ErrorEvent(message="Synthetic responder failure")

    server = RecordingServer(tmp_path, respond)
    events = await server.consume(create_thread())
    assert isinstance(events[-1], ErrorEvent)
    ((turn, outcome, _),) = server.finished
    assert turn is server.received[0]
    assert outcome is TurnOutcome.FAILED

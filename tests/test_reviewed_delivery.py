import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Literal

import httpx2
import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    OpenAIResponsesModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    StopAtTools,
    function_tool,
    output_guardrail,
)
from agents.tool_context import ToolContext
from fastapi import FastAPI, Request
from starlette.types import Message, Scope

from chatkit.agents import AgentContext, stream_agent_response
from chatkit.errors import CustomStreamError, ErrorCode
from chatkit.events import (
    ErrorEvent,
    ProgressUpdateEvent,
    ThreadCreatedEvent,
    ThreadItemDoneEvent,
    ThreadStreamEvent,
)
from chatkit.fastapi import process_request
from chatkit.models import (
    AssistantMessageItem,
    AssistantMessagePhase,
    Thread,
    UserMessageTextContent,
)
from chatkit.protocol import encode_message
from chatkit.requests import (
    ThreadAddUserMessageParams,
    ThreadGetByIdParams,
    ThreadsAddUserMessageReq,
    ThreadsGetByIdReq,
)
from chatkit.turns import ResponderEvent, Turn, TurnOutcome, UserMessageTurn
from tests._types import RequestContext
from tests.helpers.provider import client_tool_stream, mock_client, responses_stream
from tests.helpers.server import DEFAULT_CONTEXT, decode_event
from tests.helpers.turn_server import RecordingServer, create_thread


def make_reviewed_server(path: Path, agent: Agent) -> RecordingServer:
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        context = AgentContext(
            thread=turn.thread,
            store=server.store,
            request_context=turn.context,
            resume_from=None,
        )
        # This UI event is outside the final output inspected by the SDK.
        await context.stream(ProgressUpdateEvent(text="Unreviewed progress"))
        assert isinstance(turn.input, UserMessageTurn)
        (part,) = turn.input.message.content
        assert isinstance(part, UserMessageTextContent)
        result = Runner.run_streamed(
            agent,
            part.text,
            context=context,
            run_config=RunConfig(tracing_disabled=True),
        )
        async with aclosing(
            stream_agent_response(context, result, delivery="reviewed_text")
        ) as events:
            async for event in events:
                yield event

    server = RecordingServer(path, respond)
    return server


@pytest.mark.parametrize(
    "decision", ["pass", "reject", "error", "public_error", "disconnect", "no_check"]
)
async def test_reviewed_text_releases_only_approved_final_reply(
    tmp_path: Path,
    decision: Literal[
        "pass", "reject", "error", "public_error", "disconnect", "no_check"
    ],
):
    reviewing = asyncio.Event()
    allow_verdict = asyncio.Event()
    cancelled = asyncio.Event()
    sent: list[ThreadStreamEvent] = []

    @output_guardrail
    async def review(
        _context: RunContextWrapper[AgentContext], _agent: Agent, output: str
    ) -> GuardrailFunctionOutput:
        assert output == "Example reply"
        reviewing.set()
        try:
            await allow_verdict.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        if decision == "error":
            raise RuntimeError("Synthetic review failed")
        if decision == "public_error":
            raise CustomStreamError(f"Could not verify {output}", allow_retry=False)
        return GuardrailFunctionOutput(
            output_info=None, tripwire_triggered=decision == "reject"
        )

    def respond_from_provider(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=responses_stream(1, phase="final_answer"),
        )

    async with mock_client(respond_from_provider) as client:
        agent = Agent(
            name="Writer",
            model=OpenAIResponsesModel("gpt-6-luna", client),
            model_settings=ModelSettings(store=False),
            output_guardrails=[] if decision == "no_check" else [review],
        )

        server = make_reviewed_server(tmp_path, agent)
        stream = await server.open_stream(create_thread())

        async def consume() -> None:
            async with aclosing(stream):
                async for event in stream:
                    sent.append(event)

        async def reload_replies() -> list[AssistantMessageItem]:
            assert stream.thread_id is not None
            loaded = await server.process(
                ThreadsGetByIdReq(
                    params=ThreadGetByIdParams(thread_id=stream.thread_id)
                ),
                DEFAULT_CONTEXT,
            )
            assert isinstance(loaded, Thread)
            assert b"Unreviewed progress" not in encode_message(loaded)
            return [
                item
                for item in loaded.items.data
                if isinstance(item, AssistantMessageItem)
            ]

        async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
            consumer = tasks.create_task(consume())
            if decision != "no_check":
                await reviewing.wait()
                assert await reload_replies() == []
                assert not any(
                    b"Example reply" in encode_message(event) for event in sent
                )
                if decision == "disconnect":
                    consumer.cancel()
                else:
                    allow_verdict.set()

        replies = await reload_replies()
        assert not any(
            b"Unreviewed progress" in encode_message(event) for event in sent
        )
        candidate_events = [
            event for event in sent if b"Example reply" in encode_message(event)
        ]
        ((_, outcome, _),) = server.finished
        if decision == "pass":
            (reply,) = replies
            assert reply.id == "msg_1"
            assert reply.phase == "final_answer"
            assert b'"phase"' not in encode_message(reply)
            assert [part.text for part in reply.content] == ["Example reply"]
            # The model's citation metadata was not included in the reviewed final string.
            assert reply.content[0].annotations == []
            assert candidate_events == [ThreadItemDoneEvent(item=reply)]
            assert outcome is TurnOutcome.COMPLETED
        else:
            assert candidate_events == []
            assert replies == []
            if decision == "public_error":
                assert sent[-1] == ErrorEvent(
                    code=ErrorCode.STREAM_ERROR, allow_retry=False
                )
            if decision == "disconnect":
                assert cancelled.is_set()
                assert outcome is TurnOutcome.CANCELLED
            else:
                assert any(isinstance(event, ErrorEvent) for event in sent)
                assert outcome is TurnOutcome.FAILED


@pytest.mark.parametrize(
    ("initial_phase", "final_phase", "expected_phase"),
    [
        ("final_answer", None, "final_answer"),
        ("commentary", "final_answer", "final_answer"),
        ("commentary", None, None),
    ],
)
async def test_reviewed_text_preserves_its_provider_message_phase(
    tmp_path: Path,
    initial_phase: AssistantMessagePhase,
    final_phase: AssistantMessagePhase | None,
    expected_phase: AssistantMessagePhase | None,
):
    @output_guardrail
    async def approve(_context, _agent, output: str) -> GuardrailFunctionOutput:
        assert output == "Example reply"
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    payload = responses_stream(1, initial_phase=initial_phase, phase=final_phase)
    async with mock_client(
        lambda _: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=payload,
        )
    ) as client:
        server = make_reviewed_server(
            tmp_path,
            Agent(
                name="Writer",
                model=OpenAIResponsesModel("gpt-6-luna", client),
                output_guardrails=[approve],
            ),
        )
        await server.consume(create_thread())
    ((_, outcome, saved),) = server.finished
    assert outcome is TurnOutcome.COMPLETED
    (reply,) = [item for item in saved if isinstance(item, AssistantMessageItem)]
    assert reply.phase == expected_phase
    assert (reply.id == "msg_1") == (expected_phase == "final_answer")
    assert [part.text for part in reply.content] == ["Example reply"]


@pytest.mark.parametrize("finish", ["multi_step", "browser_handoff", "tool_output"])
async def test_reviewed_text_withholds_intermediate_model_items_and_browser_tools(
    tmp_path: Path, finish: Literal["multi_step", "browser_handoff", "tool_output"]
):
    @function_tool
    async def client_read(ctx: ToolContext[AgentContext], source: str) -> str:
        """Read a selection."""
        if finish == "browser_handoff":
            ctx.context.request_client_tool(
                ctx, name="read", arguments={"source": source}, continuation="local"
            )
        return "Selection"

    @function_tool
    async def server_lookup() -> str:
        """Look up a record."""
        return "Record"

    reviewed_outputs: list[str] = []

    @output_guardrail
    async def approve(_context, _agent, output: str) -> GuardrailFunctionOutput:
        reviewed_outputs.append(output)
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    if finish == "tool_output":
        # Give the model and terminal tool identical text to exercise output ownership.
        tool_payload = client_tool_stream(
            "responses",
            1,
            '{"source":"model"}',
            message_text="Selection",
            message_phase="final_answer",
        )
    else:
        tool_payload = client_tool_stream("responses", 1, '{"source":"model"}')
    payloads = iter([tool_payload, responses_stream(2)])

    def provider(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, headers={"Content-Type": "text/event-stream"}, text=next(payloads)
        )

    async with mock_client(provider) as client:
        agent = Agent(
            name="Tool example",
            model=OpenAIResponsesModel("gpt-6-luna", client),
            tools=[client_read, server_lookup],
            output_guardrails=[approve],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["client_read"])
            if finish != "multi_step"
            else "run_llm_again",
        )

        server = make_reviewed_server(tmp_path, agent)
        sent = await server.consume(create_thread())
    assert reviewed_outputs == [
        "Example reply" if finish == "multi_step" else "Selection"
    ]
    ((turn, outcome, _),) = server.finished
    loaded = await server.process(
        ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=turn.thread.id)),
        DEFAULT_CONTEXT,
    )
    assert isinstance(loaded, Thread)
    assert all(
        item.type in ("user_message", "assistant_message") for item in loaded.items.data
    )
    for public in [*sent, loaded]:
        encoded = encode_message(public)
        for private in (
            b"msg_commentary_1",
            b"check the selection",
            b"synthetic-opaque-reasoning",
            b"client_tool_call",
        ):
            assert private not in encoded
    replies = [
        item for item in loaded.items.data if isinstance(item, AssistantMessageItem)
    ]
    if finish == "browser_handoff":
        assert replies == []
        assert outcome is TurnOutcome.FAILED
    elif finish == "tool_output":
        (reply,) = replies
        assert reply.phase is None
        assert [part.text for part in reply.content] == ["Selection"]
        assert outcome is TurnOutcome.COMPLETED
    else:
        assert [(item.id, item.content[0].text) for item in replies] == [
            ("msg_2", "Example reply")
        ]
        assert outcome is TurnOutcome.COMPLETED


async def test_asgi_24_disconnect_during_output_review_allows_next_turn(tmp_path: Path):
    reviewing = asyncio.Event()
    guard_cancelled = asyncio.Event()
    reviewed: list[str] = []

    @output_guardrail
    async def review(_context, _agent, output: str) -> GuardrailFunctionOutput:
        reviewed.append(output)
        if output == "Unreviewed reply":
            reviewing.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                guard_cancelled.set()
                raise
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    def provider(request: httpx2.Request) -> httpx2.Response:
        if b"Continue" in request.content:
            payload = responses_stream(2).replace("Example reply", "Reviewed reply")
        else:
            payload = responses_stream(1).replace("Example reply", "Unreviewed reply")
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=payload
        )

    create = create_thread()
    request_body = create.model_dump_json().encode()
    initial_events: list[ThreadStreamEvent] = []

    async def receive() -> Message:
        nonlocal request_body
        if request_body:
            body, request_body = request_body, b""
            return {"type": "http.request", "body": body}
        # The provider has finished. No further writes can discover this disconnect.
        await reviewing.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        elif message.get("body"):
            initial_events.append(decode_event(message["body"]))

    async with asyncio.timeout(5), mock_client(provider) as client:
        server = make_reviewed_server(
            tmp_path,
            Agent(
                name="Writer",
                model=OpenAIResponsesModel("gpt-6-luna", client),
                output_guardrails=[review],
            ),
        )
        app = FastAPI()

        @app.post("/chatkit")
        async def chatkit(request: Request):
            return await process_request(server, request, DEFAULT_CONTEXT)

        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "POST",
            "path": "/chatkit",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        }
        await app(scope, receive, send)
        assert guard_cancelled.is_set()
        assert [outcome for _, outcome, _ in server.finished] == [TurnOutcome.CANCELLED]
        assert all(
            b"Unreviewed reply" not in encode_message(event) for event in initial_events
        )
        (thread_id,) = [
            event.thread.id
            for event in initial_events
            if isinstance(event, ThreadCreatedEvent)
        ]
        reload = ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread_id))
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as browser:

            async def reload_thread() -> Thread:
                response = await browser.post(
                    "/chatkit", json=reload.model_dump(mode="json")
                )
                response.raise_for_status()
                assert b"Unreviewed reply" not in response.content
                return Thread.model_validate_json(response.content)

            loaded = await reload_thread()
            assert [item.type for item in loaded.items.data] == ["user_message"]
            followup = ThreadsAddUserMessageReq(
                params=ThreadAddUserMessageParams(
                    thread_id=thread_id,
                    input=create.params.input.model_copy(
                        update={"content": [UserMessageTextContent(text="Continue")]}
                    ),
                )
            )
            next_turn = await browser.post(
                "/chatkit", json=followup.model_dump(mode="json")
            )
            next_turn.raise_for_status()
            next_events = [
                decode_event(frame)
                for frame in next_turn.content.split(b"\n\n")
                if frame
            ]
            (reply,) = [
                event.item
                for event in next_events
                if isinstance(event, ThreadItemDoneEvent)
                and isinstance(event.item, AssistantMessageItem)
            ]
            assert reply.id == "msg_2"
            assert [part.text for part in reply.content] == ["Reviewed reply"]
            loaded = await reload_thread()
            assert [
                item
                for item in loaded.items.data
                if isinstance(item, AssistantMessageItem)
            ] == [reply]
        assert reviewed == ["Unreviewed reply", "Reviewed reply"]
        assert [outcome for _, outcome, _ in server.finished] == [
            TurnOutcome.CANCELLED,
            TurnOutcome.COMPLETED,
        ]

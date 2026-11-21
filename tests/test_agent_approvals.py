from collections.abc import AsyncIterator
from contextlib import aclosing
from pathlib import Path

import httpx2
import pytest
from agents import (
    Agent,
    ModelSettings,
    OpenAIResponsesModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    RunState,
    function_tool,
)

from chatkit.agents import AgentContext, stream_agent_response
from chatkit.errors import CustomStreamError, ErrorCode
from chatkit.events import ErrorEvent, ThreadItemDoneEvent
from chatkit.models import AssistantMessageItem, ThreadMetadata
from chatkit.protocol import encode_message
from chatkit.requests import ThreadGetByIdParams, ThreadsGetByIdReq
from chatkit.turns import ApprovalRequired, ResponderEvent, Turn, TurnOutcome, TurnState
from tests._types import RequestContext
from tests.helpers.provider import client_tool_stream, mock_client, responses_stream
from tests.helpers.server import DEFAULT_CONTEXT
from tests.helpers.turn_server import RecordingServer, create_thread


async def test_approval_pause_saves_private_state_and_resumes_after_review(
    tmp_path: Path,
):
    calls: list[str] = []
    saved: dict[tuple[str, str, str], str] = {}

    @function_tool(needs_approval=True)
    async def client_read(source: str) -> str:
        """Read an example selection."""
        calls.append("client_read")
        return source

    @function_tool(needs_approval=True)
    async def server_lookup() -> str:
        """Look up the example value."""
        calls.append("server_lookup")
        return "found"

    class Server(RecordingServer):
        async def save_approval_checkpoint(
            self,
            thread: ThreadMetadata,
            turn_id: str,
            checkpoint: str,
            context: RequestContext,
        ) -> None:
            saved[context.user_id, thread.id, turn_id] = checkpoint

    provider_streams = iter([
        client_tool_stream("responses", 1, '{"source":"example"}'),
        responses_stream(2),
    ])

    def provider(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=next(provider_streams),
        )

    async with mock_client(provider) as client:
        agent = Agent(
            name="Example",
            model=OpenAIResponsesModel("gpt-6-luna", client),
            model_settings=ModelSettings(store=False),
            tools=[client_read, server_lookup],
        )
        run_config = RunConfig(tracing_disabled=True)

        async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
            context = AgentContext(
                thread=turn.thread,
                store=server.store,
                request_context=turn.context,
                resume_from=None,
            )
            result = Runner.run_streamed(
                agent, "Hello", context=context, run_config=run_config
            )
            async with aclosing(stream_agent_response(context, result)) as events:
                async for event in events:
                    yield event

        server = Server(tmp_path, respond)
        events = await server.consume(create_thread())
        ((turn, outcome, _),) = server.finished
        assert outcome is TurnOutcome.WAITING
        assert (
            await server.store.load_turn_state(turn.thread.id, turn.id, turn.context)
            is TurnState.WAITING
        )
        assert calls == []
        checkpoint = saved[turn.context.user_id, turn.thread.id, turn.id]
        assert "synthetic-opaque-reasoning" in checkpoint
        thread = await server.process(
            ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=turn.thread.id)),
            DEFAULT_CONTEXT,
        )
        assert isinstance(thread, ThreadMetadata)
        assert all(
            b"synthetic-opaque-reasoning" not in encode_message(item)
            for item in [*events, thread]
        )

        context = AgentContext(
            thread=turn.thread,
            store=server.store,
            request_context=turn.context,
            resume_from=None,
        )
        state = await RunState.from_string(
            agent, checkpoint, context_override=RunContextWrapper(context=context)
        )
        first, second = state.get_interruptions()
        assert [first.name, second.name] == ["client_read", "server_lookup"]
        state.approve(first)
        state.reject(second)
        resumed = Runner.run_streamed(agent, state, run_config=run_config)
        async with aclosing(stream_agent_response(context, resumed)) as events:
            replies = [
                part.text
                async for event in events
                if isinstance(event, ThreadItemDoneEvent)
                and isinstance(event.item, AssistantMessageItem)
                for part in event.item.content
            ]
        assert replies == ["Example reply"]
        assert calls == ["client_read"]
        assert resumed.interruptions == []
        assert resumed.final_output == "Example reply"


@pytest.mark.parametrize("error", ["unconfigured", "write_failed", "custom"])
async def test_turn_fails_when_approval_checkpoint_cannot_be_saved(
    tmp_path: Path, error: str
):
    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ResponderEvent]:
        yield ApprovalRequired("private-checkpoint")

    class FailingServer(RecordingServer):
        async def save_approval_checkpoint(
            self,
            thread: ThreadMetadata,
            turn_id: str,
            checkpoint: str,
            context: RequestContext,
        ) -> None:
            if error == "custom":
                raise CustomStreamError(f"Unable to save {checkpoint}")
            raise OSError("example storage failure")

    server = (RecordingServer if error == "unconfigured" else FailingServer)(
        tmp_path, respond
    )
    events = await server.consume(create_thread())
    assert events[-1] == ErrorEvent(code=ErrorCode.STREAM_ERROR, allow_retry=True)
    ((_, outcome, _),) = server.finished
    assert outcome is TurnOutcome.FAILED
    assert all(b"private-checkpoint" not in encode_message(event) for event in events)

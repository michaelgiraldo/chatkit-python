import json
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Literal
from unittest.mock import AsyncMock

import httpx2
import pytest
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    StopAtTools,
    function_tool,
)
from agents.tool_context import ToolContext

from chatkit.agent_input import client_tool_continuation_input, simple_to_agent_input
from chatkit.agents import AgentContext, stream_agent_response
from chatkit.events import (
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartTextDelta,
    ErrorEvent,
    ThreadCreatedEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
)
from chatkit.models import (
    AssistantMessageItem,
    ClientToolCallItem,
    InferenceOptions,
    Thread,
    URLSource,
    UserMessageInput,
    UserMessageTextContent,
)
from chatkit.protocol import encode_message, encode_sse, parse_request
from chatkit.requests import (
    ThreadAddUserMessageParams,
    ThreadCreateParams,
    ThreadGetByIdParams,
    ThreadsAddClientToolOutputReq,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsGetByIdReq,
)
from chatkit.turns import (
    ClientToolResponsesContinuation,
    ClientToolTurn,
    ResponderEvent,
    Turn,
    UserMessageTurn,
)
from tests._types import RequestContext
from tests.helpers.provider import (
    chat_completions_stream,
    client_tool_stream,
    mock_client,
    responses_stream,
)
from tests.helpers.server import DEFAULT_CONTEXT, make_server


def _message_input(text: str) -> UserMessageInput:
    return UserMessageInput(
        content=[UserMessageTextContent(text=text)],
        attachments=[],
        inference_options=InferenceOptions(),
    )


def _completed_reply(events: list[ThreadStreamEvent]) -> AssistantMessageItem:
    assert not [event for event in events if isinstance(event, ErrorEvent)]
    added = [
        event.item.id
        for event in events
        if isinstance(event, ThreadItemAddedEvent)
        and isinstance(event.item, AssistantMessageItem)
    ]
    updates = [
        event.item_id for event in events if isinstance(event, ThreadItemUpdatedEvent)
    ]
    (reply,) = [
        event.item
        for event in events
        if isinstance(event, ThreadItemDoneEvent)
        and isinstance(event.item, AssistantMessageItem)
    ]
    assert added == [reply.id]
    assert updates and set(updates) == {reply.id}
    return reply


async def test_responses_stream_preserves_headers_decodes_citations_and_chains_turns(
    monkeypatch: pytest.MonkeyPatch,
):
    previous_ids: list[str | None] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        assert request.headers["User-Agent"].startswith("Agents/Python ")
        assert "ChatKit/Python" not in request.headers["User-Agent"]
        assert request.headers["X-Example-Route"] == "example"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["store"] is True
        previous_ids.append(body.get("previous_response_id"))
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=responses_stream(len(previous_ids)),
        )

    async with mock_client(handle_request) as client:
        agent = Agent(
            name="Example agent",
            model=OpenAIResponsesModel("gpt-6-luna", client),
            model_settings=ModelSettings(
                store=True,
                extra_headers={"X-Example-Route": "example"},
            ),
        )
        response_id = None
        completed_ids: list[str | None] = []

        async def respond(
            turn: Turn[RequestContext],
        ) -> AsyncIterator[ResponderEvent]:
            nonlocal response_id
            assert isinstance(turn.input, UserMessageTurn)
            context = AgentContext(
                thread=turn.thread,
                store=server.store,
                request_context=turn.context,
                resume_from=None,
            )
            result = Runner.run_streamed(
                agent,
                await simple_to_agent_input(turn.input.message),
                context=context,
                run_config=RunConfig(tracing_disabled=True),
                previous_response_id=response_id,
            )
            async with aclosing(stream_agent_response(context, result)) as events:
                async for event in events:
                    yield event
            response_id = result.last_response_id
            completed_ids.append(response_id)

        with make_server(respond) as server:
            history = AsyncMock(wraps=server.store.load_thread_items)
            monkeypatch.setattr(server.store, "load_thread_items", history)
            first_turn = await server.process_streaming(
                ThreadsCreateReq(
                    params=ThreadCreateParams(
                        input=_message_input("Example question 1")
                    )
                )
            )
            thread_id = next(
                event.thread.id
                for event in first_turn
                if isinstance(event, ThreadCreatedEvent)
            )
            second_turn = await server.process_streaming(
                ThreadsAddUserMessageReq(
                    params=ThreadAddUserMessageParams(
                        thread_id=thread_id, input=_message_input("Example question 2")
                    )
                )
            )
            history.assert_not_awaited()
            loaded = await server.process_non_streaming(
                ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread_id))
            )

        for turn, events in enumerate((first_turn, second_turn), start=1):
            annotations = [
                event.update
                for event in events
                if isinstance(event, ThreadItemUpdatedEvent)
                and isinstance(event.update, AssistantMessageContentPartAnnotationAdded)
            ]
            assert len(annotations) == 1
            assert annotations[0].annotation_index == 0
            source = annotations[0].annotation.source
            assert isinstance(source, URLSource)
            assert source.url == "https://example.com/source"
            reply = _completed_reply(events)
            assert (reply.id, reply.content[0].text) == (f"msg_{turn}", "Example reply")

    assert previous_ids == [None, "resp_1"]
    assert completed_ids == ["resp_1", "resp_2"]
    assert isinstance(loaded, Thread)
    assert [
        (item.id, item.content[0].text)
        for item in loaded.items.data
        if isinstance(item, AssistantMessageItem)
    ] == [("msg_1", "Example reply"), ("msg_2", "Example reply")]


async def test_chat_completions_stream_preserves_headers_and_persists_distinct_replies():
    requested_paths: list[str] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["User-Agent"] == "ExampleApp/1.0"
        assert request.headers["X-Example-Route"] == "example"
        assert json.loads(request.content)["stream"] is True
        requested_paths.append(request.url.path)
        turn = len(requested_paths)
        choices = [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Example "},
                "finish_reason": None,
            },
            {"index": 0, "delta": {"content": f"reply {turn}"}, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ]
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=chat_completions_stream(turn, choices),
        )

    async with mock_client(handle_request) as client:
        agent = Agent(
            name="Example agent",
            model=OpenAIChatCompletionsModel("gpt-6-luna", client),
            model_settings=ModelSettings(
                extra_headers={
                    "User-Agent": "ExampleApp/1.0",
                    "X-Example-Route": "example",
                }
            ),
        )

        async def respond(
            turn: Turn[RequestContext],
        ) -> AsyncIterator[ResponderEvent]:
            assert isinstance(turn.input, UserMessageTurn)
            context = AgentContext(
                thread=turn.thread,
                store=server.store,
                request_context=turn.context,
                resume_from=None,
            )
            result = Runner.run_streamed(
                agent,
                await simple_to_agent_input(turn.input.message),
                context=context,
                run_config=RunConfig(tracing_disabled=True),
            )
            async with aclosing(stream_agent_response(context, result)) as events:
                async for event in events:
                    yield event

        with make_server(respond) as server:
            first_turn = await server.process_streaming(
                ThreadsCreateReq(
                    params=ThreadCreateParams(
                        input=_message_input("Example question 1")
                    )
                )
            )
            thread_id = next(
                event.thread.id
                for event in first_turn
                if isinstance(event, ThreadCreatedEvent)
            )
            second_turn = await server.process_streaming(
                ThreadsAddUserMessageReq(
                    params=ThreadAddUserMessageParams(
                        thread_id=thread_id, input=_message_input("Example question 2")
                    )
                )
            )
            loaded = await server.process_non_streaming(
                ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread_id))
            )

    assert requested_paths == ["/v1/chat/completions", "/v1/chat/completions"]
    first_reply = _completed_reply(first_turn)
    second_reply = _completed_reply(second_turn)
    assert first_reply.id != second_reply.id
    assert "__fake_id__" not in {first_reply.id, second_reply.id}
    for turn, events in enumerate((first_turn, second_turn), start=1):
        assert [
            event.update.delta
            for event in events
            if isinstance(event, ThreadItemUpdatedEvent)
            and isinstance(event.update, AssistantMessageContentPartTextDelta)
        ] == ["Example ", f"reply {turn}"]
    assert isinstance(loaded, Thread)
    assert [
        (item.id, item.content[0].text)
        for item in loaded.items.data
        if isinstance(item, AssistantMessageItem)
    ] == [(first_reply.id, "Example reply 1"), (second_reply.id, "Example reply 2")]


@pytest.mark.parametrize(
    "mode",
    ["local_responses", "local_chat_completions", "previous_response", "conversation"],
)
async def test_client_tool_handoff_uses_invocation_id_and_survives_reload(
    mode: Literal[
        "local_responses", "local_chat_completions", "previous_response", "conversation"
    ],
):
    api: Literal["responses", "chat_completions"] = (
        "chat_completions" if mode == "local_chat_completions" else "responses"
    )
    continuation: Literal["local", "responses"] = (
        "responses" if mode in {"previous_response", "conversation"} else "local"
    )
    conversation_id = "conv_tool" if mode == "conversation" else None
    provider_requests = 0
    server_calls: list[str] = []
    model_arguments = '{"source": "model"}'
    browser_arguments = {"kind": "selection"}
    browser_name = "read_canvas_selection"

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        nonlocal provider_requests
        provider_requests += 1
        turn = (provider_requests + 1) // 2
        expected_client_id = f"call_client_read_{turn}"
        expected_server_id = f"call_server_lookup_{turn}"
        expected_result = None if turn == 1 else {"selection": turn}
        body = json.loads(request.content)
        is_callback = provider_requests % 2 == 0
        assert request.method == "POST"
        assert request.url.path == (
            "/v1/responses" if api == "responses" else "/v1/chat/completions"
        )
        if api == "responses":
            assert body.get("store") == (continuation == "responses")
            assert body.get("conversation") == conversation_id
            assert body.get("previous_response_id") == (
                f"resp_tool_{turn}"
                if mode == "previous_response" and is_callback
                else None
            )
        if not is_callback:
            stream = client_tool_stream(api, turn, model_arguments)
        elif api == "responses":
            inputs = body["input"]
            if continuation == "responses":
                assert [item["type"] for item in inputs] == [
                    "function_call_output",
                    "function_call_output",
                ]
            else:
                assert [item["type"] for item in inputs] == [
                    "message",
                    "reasoning",
                    "message",
                    "function_call",
                    "function_call",
                    "function_call_output",
                    "function_call_output",
                ]
                user, reasoning, commentary, client_call, server_call = inputs[:5]
                assert user["role"] == "user"
                assert reasoning["encrypted_content"] == "synthetic-opaque-reasoning"
                assert commentary["phase"] == "commentary"
                assert commentary["content"][0]["text"] == "I’ll check the selection."
                assert (client_call["name"], client_call["call_id"]) == (
                    "client_read",
                    expected_client_id,
                )
                assert client_call["arguments"] == model_arguments
                assert (server_call["name"], server_call["call_id"]) == (
                    "server_lookup",
                    expected_server_id,
                )
                assert server_call["arguments"] == "{}"
            assert {item["call_id"]: item["output"] for item in inputs[-2:]} == {
                expected_client_id: json.dumps(expected_result),
                expected_server_id: "Server data",
            }
            stream = responses_stream(turn)
        else:
            messages = body["messages"]
            assert messages[0]["role"] == "user"
            assert [
                (call["id"], call["function"]["name"], call["function"]["arguments"])
                for message in messages
                for call in message.get("tool_calls", [])
            ] == [
                (expected_client_id, "client_read", model_arguments),
                (expected_server_id, "server_lookup", "{}"),
            ]
            outputs = [message for message in messages if message["role"] == "tool"]
            assert len(outputs) == 2
            assert {
                message["tool_call_id"]: message["content"] for message in outputs
            } == {
                expected_client_id: json.dumps(expected_result),
                expected_server_id: "Server data",
            }
            stream = chat_completions_stream(
                turn,
                [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Example reply"},
                        "finish_reason": None,
                    },
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                ],
            )
        return httpx2.Response(
            200, headers={"Content-Type": "text/event-stream"}, text=stream
        )

    @function_tool
    async def client_read(
        ctx: ToolContext[AgentContext[RequestContext]], source: str
    ) -> None:
        """Read a selection from the browser."""
        assert source == "model"
        ctx.context.request_client_tool(
            ctx,
            name=browser_name,
            arguments=browser_arguments,
            continuation=continuation,
        )

    @function_tool
    async def server_lookup(ctx: ToolContext[AgentContext[RequestContext]]) -> str:
        """Read data from the server."""
        server_calls.append(ctx.tool_call_id)
        return "Server data"

    pending_items: list[ClientToolCallItem] = []
    async with mock_client(handle_request) as client:
        model_type = (
            OpenAIResponsesModel if api == "responses" else OpenAIChatCompletionsModel
        )
        agent = Agent[AgentContext[RequestContext]](
            name="Tool example",
            model=model_type("gpt-6-luna", client),
            model_settings=ModelSettings(
                store=continuation == "responses" if api == "responses" else None
            ),
            tools=[client_read, server_lookup],
            tool_use_behavior=StopAtTools(stop_at_tool_names=[client_read.name]),
        )

        async def respond(
            turn: Turn[RequestContext],
        ) -> AsyncIterator[ResponderEvent]:
            context = AgentContext(
                thread=turn.thread,
                store=server.store,
                request_context=turn.context,
                resume_from=None,
            )
            previous_response_id = None
            if isinstance(turn.input, ClientToolTurn):
                assert turn.input.invocation.continuation_turn_id == turn.id
                input_items = client_tool_continuation_input(turn.input)
                if mode == "previous_response":
                    saved = turn.input.invocation.model_continuation
                    assert isinstance(saved, ClientToolResponsesContinuation)
                    previous_response_id = saved.response_id
            else:
                assert isinstance(turn.input, UserMessageTurn)
                input_items = await simple_to_agent_input(turn.input.message)
            result = Runner.run_streamed(
                agent,
                input_items,
                context=context,
                run_config=RunConfig(tracing_disabled=True),
                conversation_id=conversation_id,
                previous_response_id=previous_response_id,
            )
            async with aclosing(stream_agent_response(context, result)) as events:
                async for event in events:
                    yield event

        with make_server(respond) as server:
            events = await server.process_streaming(
                ThreadsCreateReq(
                    params=ThreadCreateParams(input=_message_input("Example question"))
                )
            )
            thread_id = next(
                event.thread.id
                for event in events
                if isinstance(event, ThreadCreatedEvent)
            )
            for turn in (1, 2):
                if turn == 2:
                    events = await server.process_streaming(
                        ThreadsAddUserMessageReq(
                            params=ThreadAddUserMessageParams(
                                thread_id=thread_id,
                                input=_message_input("Another question"),
                            )
                        )
                    )
                assert not any(isinstance(event, ErrorEvent) for event in events)
                (pending_event,) = [
                    event
                    for event in events
                    if isinstance(event, ThreadItemDoneEvent)
                    and isinstance(event.item, ClientToolCallItem)
                ]
                pending = pending_event.item
                assert isinstance(pending, ClientToolCallItem)
                assert (pending.name, pending.status) == (browser_name, "pending")
                invocation = await server.store.load_client_tool_invocation(
                    thread_id, pending.id, DEFAULT_CONTEXT
                )
                assert invocation.call_id == f"call_client_read_{turn}"
                assert invocation.model_continuation is not None
                pending_items.append(pending)
                expected_arguments = {"item_id": pending.id, "input": browser_arguments}
                assert pending.arguments == browser_arguments
                frame = json.loads(encode_sse(pending_event).removeprefix(b"data: "))
                assert frame["item"]["arguments"] == expected_arguments
                for public_item in (
                    pending.model_dump(),
                    pending_event.model_dump()["item"],
                    frame["item"],
                ):
                    assert {"model_continuation", "call_id"}.isdisjoint(public_item)

                loaded_pending = await server.process_non_streaming(
                    ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread_id))
                )
                assert isinstance(loaded_pending, Thread)
                (pending_from_history,) = [
                    item
                    for item in json.loads(encode_message(loaded_pending))["items"][
                        "data"
                    ]
                    if item["id"] == pending.id
                ]
                assert pending_from_history["arguments"] == expected_arguments
                assert {"model_continuation", "call_id"}.isdisjoint(
                    pending_from_history
                )

                callback = parse_request(
                    json.dumps({
                        "type": "threads.add_client_tool_output",
                        "params": {
                            "thread_id": thread_id,
                            "result": {
                                "item_id": frame["item"]["arguments"]["item_id"],
                                "output": None if turn == 1 else {"selection": turn},
                            },
                        },
                    })
                )
                assert isinstance(callback, ThreadsAddClientToolOutputReq)
                resumed = await server.process_streaming(callback)
                assert _completed_reply(resumed).content[0].text == "Example reply"
            loaded = await server.process_non_streaming(
                ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread_id))
            )

    assert provider_requests == 4
    assert server_calls == ["call_server_lookup_1", "call_server_lookup_2"]
    ids = [item.id for item in pending_items]
    if api == "responses":
        assert ids == ["fc_client_read_1", "fc_client_read_2"]
    else:
        assert ids[0] != ids[1]
        assert all(item_id.startswith("tc_") for item_id in ids)
    assert isinstance(loaded, Thread)
    assert [
        (item.id, item.status, item.output)
        for item in loaded.items.data
        if isinstance(item, ClientToolCallItem)
    ] == [
        (
            item.id,
            "completed",
            None if index == 1 else {"selection": index},
        )
        for index, item in enumerate(pending_items, 1)
    ]
    browser_items = [
        item
        for item in json.loads(encode_message(loaded))["items"]["data"]
        if item["type"] == "client_tool_call"
    ]
    assert [item["arguments"] for item in browser_items] == [
        browser_arguments,
        browser_arguments,
    ]
    assert all(
        {"model_continuation", "call_id"}.isdisjoint(item) for item in browser_items
    )

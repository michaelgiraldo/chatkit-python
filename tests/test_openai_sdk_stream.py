import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Literal

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
from openai import AsyncOpenAI

from chatkit.agent_input import simple_to_agent_input
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
from chatkit.requests import (
    ThreadAddClientToolOutputParams,
    ThreadAddUserMessageParams,
    ThreadCreateParams,
    ThreadGetByIdParams,
    ThreadsAddClientToolOutputReq,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsGetByIdReq,
)
from chatkit.turns import ClientToolTurn, Turn, UserMessageTurn
from tests._types import RequestContext
from tests.helpers.server import make_server


def _responses_stream(turn: int) -> str:
    citation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 7,
        "title": "Example",
        "url": "https://example.com/source",
    }
    item = {
        "id": f"msg_{turn}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "Example reply", "annotations": [citation]}
        ],
    }
    response = {
        "id": f"resp_{turn}",
        "created_at": 1,
        "model": "gpt-6-luna",
        "object": "response",
        "output": [item],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "item_id": item["id"],
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": item["id"],
            "content_index": 0,
            "delta": "Example reply",
            "logprobs": [],
        },
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "item_id": item["id"],
            "content_index": 0,
            "text": "Example reply",
            "logprobs": [],
        },
        *(
            {
                "type": "response.output_text.annotation.added",
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "annotation_index": index,
                "annotation": annotation,
            }
            for index, annotation in enumerate((None, citation))
        ),
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]
    return "".join(
        f"data: {json.dumps(dict(event, sequence_number=index))}\n\n"
        for index, event in enumerate(events)
    )


def _mock_client(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://example.com/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


def _chat_completions_stream(turn: int, choices: list[dict]) -> str:
    chunk = {
        "id": f"chatcmpl_example_{turn}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-6-luna",
    }
    return (
        "".join(
            f"data: {json.dumps(dict(chunk, choices=[choice]))}\n\n"
            for choice in choices
        )
        + "data: [DONE]\n\n"
    )


def _client_tool_stream(
    api: Literal["responses", "chat_completions"], turn: int
) -> str:
    names = ("client_read", "server_lookup")
    if api == "chat_completions":
        calls = [
            {
                "index": index,
                "id": f"call_{name}_{turn}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for index, name in enumerate(names)
        ]
        return _chat_completions_stream(
            turn,
            [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "tool_calls": calls},
                    "finish_reason": None,
                },
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
            ],
        )

    items = [
        {
            "id": f"fc_{name}_{turn}",
            "type": "function_call",
            "call_id": f"call_{name}_{turn}",
            "name": name,
            "arguments": "{}",
            "status": "completed",
        }
        for name in names
    ]
    events = [
        {
            "type": f"response.output_item.{stage}",
            "output_index": index,
            "item": item,
        }
        for index, item in enumerate(items)
        for stage in ("added", "done")
    ]
    events.append({
        "type": "response.completed",
        "response": {
            "id": f"resp_tool_{turn}",
            "created_at": 1,
            "model": "gpt-6-luna",
            "object": "response",
            "output": items,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
        },
    })
    return "".join(
        f"data: {json.dumps(dict(event, sequence_number=index))}\n\n"
        for index, event in enumerate(events)
    )


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


async def test_responses_stream_preserves_headers_decodes_citations_and_chains_turns():
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
            text=_responses_stream(len(previous_ids)),
        )

    async with _mock_client(handle_request) as client:
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
        ) -> AsyncIterator[ThreadStreamEvent]:
            nonlocal response_id
            assert isinstance(turn.input, UserMessageTurn)
            context = AgentContext(
                thread=turn.thread, store=server.store, request_context=turn.context
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
            text=_chat_completions_stream(turn, choices),
        )

    async with _mock_client(handle_request) as client:
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
        ) -> AsyncIterator[ThreadStreamEvent]:
            assert isinstance(turn.input, UserMessageTurn)
            context = AgentContext(
                thread=turn.thread, store=server.store, request_context=turn.context
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


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
async def test_client_tool_handoff_uses_invocation_id_and_survives_reload(
    api: Literal["responses", "chat_completions"],
):
    provider_requests = 0
    server_calls: list[str] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        nonlocal provider_requests
        provider_requests += 1
        turn = (provider_requests + 1) // 2
        expected_call_id = f"call_client_read_{turn}"
        expected_result = f"selection-{turn}"
        body = json.loads(request.content)
        assert request.url.path == (
            "/v1/responses" if api == "responses" else "/v1/chat/completions"
        )
        if provider_requests % 2:
            stream = _client_tool_stream(api, turn)
        elif api == "responses":
            call, output = body["input"]
            assert (call["name"], call["call_id"]) == ("client_read", expected_call_id)
            assert output["call_id"] == expected_call_id
            assert json.loads(output["output"]) == expected_result
            stream = _responses_stream(turn)
        else:
            call, output = body["messages"]
            (invocation,) = call["tool_calls"]
            assert (invocation["function"]["name"], invocation["id"]) == (
                "client_read",
                expected_call_id,
            )
            assert output["tool_call_id"] == expected_call_id
            assert json.loads(output["content"]) == expected_result
            stream = _chat_completions_stream(
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
    async def client_read(ctx: ToolContext[AgentContext[RequestContext]]) -> None:
        """Read a selection from the browser."""
        ctx.context.request_client_tool(ctx, name="client_read", arguments={})

    @function_tool
    async def server_lookup(ctx: ToolContext[AgentContext[RequestContext]]) -> str:
        """Read data from the server."""
        server_calls.append(ctx.tool_call_id)
        return "Server data"

    pending_items: list[ClientToolCallItem] = []
    async with _mock_client(handle_request) as client:
        model_type = (
            OpenAIResponsesModel if api == "responses" else OpenAIChatCompletionsModel
        )
        agent = Agent[AgentContext[RequestContext]](
            name="Tool example",
            model=model_type("gpt-6-luna", client),
            tools=[client_read, server_lookup],
            tool_use_behavior=StopAtTools(stop_at_tool_names=[client_read.name]),
        )

        async def respond(
            turn: Turn[RequestContext],
        ) -> AsyncIterator[ThreadStreamEvent]:
            context = AgentContext(
                thread=turn.thread, store=server.store, request_context=turn.context
            )
            if isinstance(turn.input, ClientToolTurn):
                input_item = turn.input.tool_call
            else:
                assert isinstance(turn.input, UserMessageTurn)
                input_item = turn.input.message
            result = Runner.run_streamed(
                agent,
                await simple_to_agent_input(input_item),
                context=context,
                run_config=RunConfig(tracing_disabled=True),
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
                (pending,) = [
                    event.item
                    for event in events
                    if isinstance(event, ThreadItemDoneEvent)
                    and isinstance(event.item, ClientToolCallItem)
                ]
                assert (pending.name, pending.call_id, pending.status) == (
                    "client_read",
                    f"call_client_read_{turn}",
                    "pending",
                )
                pending_items.append(pending)
                resumed = await server.process_streaming(
                    ThreadsAddClientToolOutputReq(
                        params=ThreadAddClientToolOutputParams(
                            thread_id=thread_id, result=f"selection-{turn}"
                        )
                    )
                )
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
        (item.id, item.call_id, item.status, item.output)
        for item in loaded.items.data
        if isinstance(item, ClientToolCallItem)
    ] == [
        (item.id, item.call_id, "completed", f"selection-{index}")
        for index, item in enumerate(pending_items, 1)
    ]

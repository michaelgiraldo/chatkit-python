import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing

import httpx2
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
)
from openai import AsyncOpenAI

from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.turns import Turn, UserMessageTurn
from chatkit.types import (
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartTextDelta,
    AssistantMessageItem,
    ErrorEvent,
    InferenceOptions,
    Thread,
    ThreadAddUserMessageParams,
    ThreadCreatedEvent,
    ThreadCreateParams,
    ThreadGetByIdParams,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsGetByIdReq,
    ThreadStreamEvent,
    URLSource,
    UserMessageInput,
    UserMessageTextContent,
)
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
    assert [
        (item.id, item.content[0].text)
        for item in Thread.model_validate_json(loaded.json).items.data
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
        chunk = {
            "id": f"chatcmpl_example_{turn}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-6-luna",
        }
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
            text="".join(
                f"data: {json.dumps(dict(chunk, choices=[choice]))}\n\n"
                for choice in choices
            )
            + "data: [DONE]\n\n",
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
    assert [
        (item.id, item.content[0].text)
        for item in Thread.model_validate_json(loaded.json).items.data
        if isinstance(item, AssistantMessageItem)
    ] == [(first_reply.id, "Example reply 1"), (second_reply.id, "Example reply 2")]

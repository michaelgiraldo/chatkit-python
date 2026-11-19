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
from chatkit.types import (
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageItem,
    InferenceOptions,
    ThreadAddUserMessageParams,
    ThreadCreatedEvent,
    ThreadCreateParams,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadStreamEvent,
    URLSource,
    UserMessageInput,
    UserMessageItem,
    UserMessageTextContent,
)
from tests._types import RequestContext
from tests.test_chatkit_server import make_server


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
            thread: ThreadMetadata,
            message: UserMessageItem | None,
            request_context: RequestContext,
        ) -> AsyncIterator[ThreadStreamEvent]:
            nonlocal response_id
            assert message is not None
            context = AgentContext(
                thread=thread, store=server.store, request_context=request_context
            )
            result = Runner.run_streamed(
                agent,
                await simple_to_agent_input(message),
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
            assert [
                (event.item.id, event.item.content[0].text)
                for event in events
                if isinstance(event, ThreadItemDoneEvent)
                and isinstance(event.item, AssistantMessageItem)
            ] == [(f"msg_{turn}", "Example reply")]

    assert previous_ids == [None, "resp_1"]
    assert completed_ids == ["resp_1", "resp_2"]


async def test_chat_completions_stream_preserves_application_headers():
    requested_paths: list[str] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["User-Agent"] == "ExampleApp/1.0"
        assert request.headers["X-Example-Route"] == "example"
        assert json.loads(request.content)["stream"] is True
        requested_paths.append(request.url.path)
        reply = {
            "id": "chatcmpl_example",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-6-luna",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Example reply"},
                    "finish_reason": "stop",
                },
            ],
        }
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"data: {json.dumps(reply)}\n\ndata: [DONE]\n\n",
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
            thread: ThreadMetadata,
            message: UserMessageItem | None,
            request_context: RequestContext,
        ) -> AsyncIterator[ThreadStreamEvent]:
            assert message is not None
            context = AgentContext(
                thread=thread, store=server.store, request_context=request_context
            )
            result = Runner.run_streamed(
                agent,
                await simple_to_agent_input(message),
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

    assert requested_paths == ["/v1/chat/completions"]
    assert [
        event.item.content[0].text
        for event in events
        if isinstance(event, ThreadItemDoneEvent)
        and isinstance(event.item, AssistantMessageItem)
    ] == ["Example reply"]

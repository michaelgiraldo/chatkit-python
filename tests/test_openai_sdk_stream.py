import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import httpx2
from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig, Runner
from openai import AsyncOpenAI

from chatkit.agents import AgentContext, stream_agent_response
from chatkit.server import agents_sdk_user_agent_override
from chatkit.types import (
    AssistantMessageContentPartAnnotationAdded,
    Page,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    URLSource,
)


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


async def test_openai_sdk_stream_decodes_citations_and_chains_completed_turns():
    previous_ids: list[str | None] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/responses"
        assert " ChatKit/Python " in request.headers["User-Agent"]
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["store"] is True
        previous_ids.append(body.get("previous_response_id"))
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_responses_stream(len(previous_ids)),
        )

    thread = ThreadMetadata(id="example", created_at=datetime.now(UTC))
    store = Mock()
    store.load_thread_items = AsyncMock(return_value=Page())
    config = RunConfig(tracing_disabled=True, model_settings=ModelSettings(store=True))
    transport = httpx2.MockTransport(handle_request)

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="https://example.com/v1",
        http_client=httpx2.AsyncClient(transport=transport),
        max_retries=0,
    ) as client:
        agent = Agent(
            name="Example agent", model=OpenAIResponsesModel("gpt-6-luna", client)
        )
        response_id = None
        for turn in (1, 2):
            context = AgentContext(thread=thread, store=store, request_context=None)
            with agents_sdk_user_agent_override():
                result = Runner.run_streamed(
                    agent,
                    f"Example question {turn}",
                    context=context,
                    run_config=config,
                    previous_response_id=response_id,
                )
                events = [
                    event async for event in stream_agent_response(context, result)
                ]

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
            assert any(
                isinstance(event, ThreadItemDoneEvent)
                and event.item.id == f"msg_{turn}"
                for event in events
            )
            response_id = result.last_response_id
            assert response_id == f"resp_{turn}"

    assert previous_ids == [None, "resp_1"]

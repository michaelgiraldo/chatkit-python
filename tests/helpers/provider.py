"""Synthetic model HTTP responses for SDK integration tests."""

import json
from collections.abc import Callable
from typing import Literal

import httpx2
from openai import AsyncOpenAI


def responses_stream(
    turn: int,
    *,
    initial_phase: Literal["commentary", "final_answer"] | None = None,
    phase: Literal["commentary", "final_answer"] | None = None,
) -> str:
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
        **({"phase": phase} if phase is not None else {}),
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
            "item": {
                **item,
                "phase": initial_phase,
                "status": "in_progress",
                "content": [],
            },
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


def mock_client(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://example.com/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


def chat_completions_stream(turn: int, choices: list[dict]) -> str:
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


def client_tool_stream(
    api: Literal["responses", "chat_completions"],
    turn: int,
    model_arguments: str,
    *,
    message_text: str = "I’ll check the selection.",
    message_phase: Literal["commentary", "final_answer"] = "commentary",
) -> str:
    names = ("client_read", "server_lookup")
    if api == "chat_completions":
        calls = [
            {
                "index": index,
                "id": f"call_{name}_{turn}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": model_arguments if name == "client_read" else "{}",
                },
            }
            for index, name in enumerate(names)
        ]
        return chat_completions_stream(
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

    commentary_content = {
        "type": "output_text",
        "text": message_text,
        "annotations": [],
    }
    items = [
        {
            "id": f"rs_tool_{turn}",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "synthetic-opaque-reasoning",
            "status": "completed",
        },
        {
            "id": f"msg_commentary_{turn}",
            "type": "message",
            "role": "assistant",
            "phase": message_phase,
            "content": [commentary_content],
            "status": "completed",
        },
        *[
            {
                "id": f"fc_{name}_{turn}",
                "type": "function_call",
                "call_id": f"call_{name}_{turn}",
                "name": name,
                "arguments": model_arguments if name == "client_read" else "{}",
                "status": "completed",
            }
            for name in names
        ],
    ]
    events = [
        {
            "type": f"response.output_item.{stage}",
            "output_index": index,
            "item": (
                {**item, "content": [], "status": "in_progress"}
                if stage == "added" and item["type"] == "message"
                else item
            ),
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

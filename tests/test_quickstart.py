from fastapi.testclient import TestClient

from chatkit.events import (
    AssistantMessageContentPartTextDelta,
    ThreadCreatedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
)
from chatkit.models import AssistantMessageItem
from docs.examples import quickstart
from docs.examples.quickstart import app, server
from tests.helpers.server import decode_event


def _message(text: str) -> dict:
    return {
        "content": [{"type": "input_text", "text": text}],
        "attachments": [],
        "inference_options": {},
    }


def _events(body: bytes) -> list[ThreadStreamEvent]:
    return [
        decode_event(line) for line in body.splitlines() if line.startswith(b"data: ")
    ]


def test_quickstart_runs_two_turns_and_reloads_without_attachment_storage(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert server.attachment_metadata_store is None

    with TestClient(app) as client:
        first = client.post(
            "/chatkit",
            json={"type": "threads.create", "params": {"input": _message("Hi")}},
        )
        assert first.status_code == 200
        assert first.headers["content-type"].startswith("text/event-stream")
        first_events = _events(first.content)
        thread = next(
            e.thread for e in first_events if isinstance(e, ThreadCreatedEvent)
        )
        first_reply = first_events[-1]
        assert isinstance(first_reply, ThreadItemDoneEvent)
        assert first_reply.item.type == "assistant_message"

        follow_up = client.post(
            "/chatkit",
            json={
                "type": "threads.add_user_message",
                "params": {"thread_id": thread.id, "input": _message("Again")},
            },
        )
        assert follow_up.status_code == 200
        second_events = _events(follow_up.content)
        second_reply = second_events[-1]
        assert isinstance(second_reply, ThreadItemDoneEvent)
        assert second_reply.item.type == "assistant_message"
        assert second_reply.item.id != first_reply.item.id

        reloaded = client.post(
            "/chatkit",
            json={"type": "threads.get_by_id", "params": {"thread_id": thread.id}},
        )
        assert reloaded.status_code == 200
        items = reloaded.json()["items"]["data"]
        assert [item["type"] for item in items] == [
            "user_message",
            "assistant_message",
            "user_message",
            "assistant_message",
        ]
        assert [items[1]["id"], items[3]["id"]] == [
            first_reply.item.id,
            second_reply.item.id,
        ]
        assert items[3]["content"][0]["text"] == "Hello, world!"


def test_quickstart_serves_the_official_client_and_saves_widget_actions():
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "<openai-chatkit>" in page.text
        assert 'new URL("/chatkit", window.location.href)' in page.text
        assert (
            "https://cdn.platform.openai.com/deployments/chatkit/chatkit.js"
            in page.text
        )

        streamed = client.post(
            "/chatkit",
            json={"type": "threads.create", "params": {"input": _message("widget")}},
        )
        assert streamed.status_code == 200
        events = _events(streamed.content)
        created, widget = events[0], events[-1]
        assert isinstance(created, ThreadCreatedEvent)
        assert isinstance(widget, ThreadItemDoneEvent)
        assert widget.item.type == "widget"
        action = widget.item.widget.model_dump()["children"][1]["onClickAction"]
        response = client.post(
            "/chatkit",
            json={
                "type": "threads.custom_action",
                "params": {
                    "thread_id": created.thread.id,
                    "item_id": widget.item.id,
                    "action": action,
                },
            },
        )
        assert response.status_code == 200
        updated = _events(response.content)[-1]
        assert isinstance(updated, ThreadItemReplacedEvent)
        assert updated.item.id == widget.item.id

        reloaded = client.post(
            "/chatkit",
            json={
                "type": "threads.get_by_id",
                "params": {"thread_id": created.thread.id},
            },
        )
        assert reloaded.status_code == 200
        assert reloaded.json()["items"]["data"][-1]["widget"]["children"] == [
            {"type": "Text", "value": "Widget state saved."}
        ]


def test_quickstart_slow_reply_streams_text_before_finishing(monkeypatch):
    monkeypatch.setattr(quickstart, "_SLOW_STEP_SECONDS", 0)
    with TestClient(app) as client:
        response = client.post(
            "/chatkit",
            json={"type": "threads.create", "params": {"input": _message("slow")}},
        )
        assert response.status_code == 200

    events = _events(response.content)
    chunks = [
        event.update.delta
        for event in events
        if isinstance(event, ThreadItemUpdatedEvent)
        and isinstance(event.update, AssistantMessageContentPartTextDelta)
    ]
    assert chunks[0] == "The demo is streaming."
    assert len(chunks) > 1
    reply = events[-1]
    assert isinstance(reply, ThreadItemDoneEvent)
    assert isinstance(reply.item, AssistantMessageItem)
    assert reply.item.content[0].text == "".join(chunks)

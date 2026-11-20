from fastapi.testclient import TestClient

from chatkit.types import ThreadCreatedEvent, ThreadItemDoneEvent
from docs.examples.quickstart import app, server
from tests.helpers.server import decode_event


def test_quickstart_runs_two_turns_and_reloads_without_attachment_storage(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert server.attachment_metadata_store is None

    def message(text: str) -> dict:
        return {
            "content": [{"type": "input_text", "text": text}],
            "attachments": [],
            "inference_options": {},
        }

    with TestClient(app) as client:
        first = client.post(
            "/chatkit",
            json={"type": "threads.create", "params": {"input": message("Hi")}},
        )
        assert first.status_code == 200
        assert first.headers["content-type"].startswith("text/event-stream")
        first_events = [
            decode_event(line)
            for line in first.content.splitlines()
            if line.startswith(b"data: ")
        ]
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
                "params": {"thread_id": thread.id, "input": message("Again")},
            },
        )
        assert follow_up.status_code == 200
        second_events = [
            decode_event(line)
            for line in follow_up.content.splitlines()
            if line.startswith(b"data: ")
        ]
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

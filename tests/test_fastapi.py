from collections.abc import Iterator

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chatkit.events import ProgressUpdateEvent, ThreadCreatedEvent
from chatkit.fastapi import process_request
from chatkit.models import ThreadMetadata
from chatkit.protocol import encode_sse
from chatkit.server import ChatKitServer
from tests._types import RequestContext
from tests.helpers.server import DEFAULT_CONTEXT, decode_event, make_server


def make_client(server: ChatKitServer[RequestContext]) -> TestClient:
    app = FastAPI()

    @app.post("/chatkit")
    async def chatkit(request: Request) -> Response:
        return await process_request(server, request, DEFAULT_CONTEXT)

    return TestClient(app)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_server() as server, make_client(server) as test_client:
        yield test_client


def test_process_request_streams_and_returns_json(client: TestClient):
    streamed = client.post(
        "/chatkit",
        json={
            "type": "threads.create",
            "params": {
                "input": {
                    "content": [{"type": "input_text", "text": "Hello"}],
                    "attachments": [],
                    "inference_options": {},
                }
            },
        },
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    first_event = decode_event(streamed.content.splitlines()[0])
    assert isinstance(first_event, ThreadCreatedEvent)

    loaded = client.post(
        "/chatkit",
        json={
            "type": "threads.get_by_id",
            "params": {"thread_id": first_event.thread.id},
        },
    )
    assert loaded.status_code == 200
    assert loaded.headers["content-type"] == "application/json"
    assert loaded.json()["id"] == first_event.thread.id


def test_process_request_rejects_malformed_json(client: TestClient):
    invalid_json = '{"type": "example-private-input"'
    response = client.post(
        "/chatkit", content=invalid_json, headers={"content-type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json()["detail"]
    assert "example-private-input" not in response.text


def test_sse_frame_decoder_keeps_data_prefix_inside_event_text():
    event = ProgressUpdateEvent(text="Received data: example")
    assert decode_event(encode_sse(event)) == event


def test_process_request_reports_invalid_fields_without_echoing_values(
    client: TestClient,
):
    response = client.post(
        "/chatkit",
        json={
            "type": "threads.get_by_id",
            "params": {"thread_id": {"value": "example-private-input"}},
        },
    )

    assert response.status_code == 422
    (issue,) = response.json()["detail"]
    assert issue["location"][-2:] == ["params", "thread_id"]
    assert issue["type"] == "string_type"
    assert issue["message"]
    assert "example-private-input" not in response.text


def test_process_request_reports_unknown_operation_without_echoing_value(
    client: TestClient,
):
    response = client.post(
        "/chatkit", json={"type": "example-private-input", "params": {}}
    )

    assert response.status_code == 422
    (issue,) = response.json()["detail"]
    assert "operation" in issue["message"]
    assert "example-private-input" not in response.text


def test_process_request_reports_missing_operation(client: TestClient):
    response = client.post("/chatkit", json={})

    assert response.status_code == 422
    (issue,) = response.json()["detail"]
    assert issue["location"] == ["type"]
    assert issue["type"] == "union_tag_not_found"


@pytest.mark.parametrize(
    "operation", ["threads.custom_action", "threads.sync_custom_action"]
)
def test_process_request_rejects_missing_action_name(
    client: TestClient, operation: str
):
    response = client.post(
        "/chatkit",
        json={
            "type": operation,
            "params": {"thread_id": "example-thread", "action": {}},
        },
    )

    assert response.status_code == 422
    (issue,) = response.json()["detail"]
    assert issue["location"][-3:] == ["params", "action", "type"]
    assert issue["type"] == "missing"


def test_process_request_keeps_invalid_answer_keys_out_of_response(client: TestClient):
    response = client.post(
        "/chatkit",
        json={
            "type": "threads.add_structured_input",
            "params": {
                "thread_id": "example-thread",
                "item_id": "example-item",
                "input": {"answers": {"example-private-question-id": "invalid"}},
            },
        },
    )

    assert response.status_code == 422
    (issue,) = response.json()["detail"]
    assert issue["location"][-3:] == ["params", "input", "answers"]
    assert "example-private-question-id" not in response.text


def test_process_request_reports_missing_thread(client: TestClient):
    response = client.post(
        "/chatkit",
        json={"type": "threads.get_by_id", "params": {"thread_id": "example-missing"}},
    )

    assert response.status_code == 404
    assert response.json()["detail"]
    assert "example-missing" not in response.text


def test_streaming_request_resolves_thread_before_sending_headers(client: TestClient):
    response = client.post(
        "/chatkit",
        json={
            "type": "threads.add_user_message",
            "params": {
                "thread_id": "example-missing",
                "input": {
                    "content": [{"type": "input_text", "text": "Hello"}],
                    "attachments": [],
                    "inference_options": {},
                },
            },
        },
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "ChatKit resource not found."}


def test_tool_output_requires_a_pending_call_before_sending_headers(client: TestClient):
    created = client.post(
        "/chatkit",
        json={
            "type": "threads.create",
            "params": {
                "input": {
                    "content": [{"type": "input_text", "text": "Hello"}],
                    "attachments": [],
                    "inference_options": {},
                },
            },
        },
    )
    event = decode_event(created.content.splitlines()[0])
    assert isinstance(event, ThreadCreatedEvent)
    response = client.post(
        "/chatkit",
        json={
            "type": "threads.add_client_tool_output",
            "params": {"thread_id": event.thread.id, "result": "example"},
        },
    )
    assert response.status_code == 422
    assert (
        response.json()["detail"]
        == "The thread is not waiting for a client tool result."
    )


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Synthetic store failure"),
        ValidationError.from_exception_data(
            "Synthetic store record",
            [{"type": "missing", "loc": ("created_at",), "input": {}}],
        ),
    ],
)
def test_process_request_propagates_unexpected_store_failure(monkeypatch, failure):
    async def fail_to_load_thread(
        thread_id: str, context: RequestContext
    ) -> ThreadMetadata:
        raise failure

    with make_server() as server, make_client(server) as client:
        monkeypatch.setattr(server.store, "load_thread", fail_to_load_thread)
        with pytest.raises(type(failure)):
            client.post(
                "/chatkit",
                json={
                    "type": "threads.get_by_id",
                    "params": {"thread_id": "example-thread"},
                },
            )

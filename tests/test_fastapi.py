from collections.abc import Iterator

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chatkit.events import (
    ProgressUpdateEvent,
    SyncCustomActionResponse,
    ThreadCreatedEvent,
)
from chatkit.fastapi import ChatKitStreamingResponse, process_request
from chatkit.models import (
    HiddenContextItem,
    Page,
    SDKHiddenContextItem,
    StructuredInputItem,
    ThreadItem,
    ThreadMetadata,
    WidgetItem,
)
from chatkit.protocol import encode_sse
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.turns import TurnOutcome, TurnState
from chatkit.widgets import Card, Text
from tests._types import RequestContext
from tests.helpers.server import DEFAULT_CONTEXT, decode_event, make_server
from tests.helpers.storage_contract import make_thread


def make_client(
    server: ChatKitServer[RequestContext], *, raise_server_exceptions: bool = True
) -> TestClient:
    app = FastAPI()

    @app.post("/chatkit")
    async def chatkit(request: Request) -> Response:
        return await process_request(server, request, DEFAULT_CONTEXT)

    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


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


@pytest.mark.parametrize(
    "reply", ["widget", "empty", "hidden_context_item", "sdk_hidden_context"]
)
async def test_sync_widget_reply_visibility(reply: str):
    thread = make_thread()
    sender = WidgetItem(
        id="widget",
        thread_id=thread.id,
        created_at=thread.created_at,
        widget=Card(children=[Text(value="Ready")]).build(),
    )
    updated = sender.model_copy(
        update={"widget": Card(children=[Text(value="Done")]).build()}
    )
    replies = {
        "widget": updated,
        "empty": None,
        "hidden_context_item": HiddenContextItem(
            id="app-note",
            thread_id=thread.id,
            created_at=thread.created_at,
            content="App only",
        ),
        "sdk_hidden_context": SDKHiddenContextItem(
            id="sdk-note",
            thread_id=thread.id,
            created_at=thread.created_at,
            content="SDK only",
        ),
    }
    result = SyncCustomActionResponse(updated_item=replies[reply])
    with (
        make_server(sync_action_callback=lambda *_: result) as server,
        make_client(server) as client,
    ):
        await server.store.save_thread(thread, DEFAULT_CONTEXT)
        await server.store.add_thread_item(thread.id, sender, DEFAULT_CONTEXT)
        response = client.post(
            "/chatkit",
            json={
                "type": "threads.sync_custom_action",
                "params": {
                    "thread_id": thread.id,
                    "item_id": sender.id,
                    "action": {"type": "confirm"},
                },
            },
        )

    assert response.status_code == 200
    if reply == "widget":
        assert response.json() == {
            "updated_item": updated.model_dump(mode="json", exclude_none=True)
        }
    else:
        assert response.json() == {}


def test_process_request_uses_app_response_for_streaming_only():
    app = FastAPI()

    def streaming_response(result: StreamingResult) -> Response:
        response = ChatKitStreamingResponse(result)
        response.headers["cache-control"] = "no-store"
        return response

    with make_server() as server:

        @app.post("/chatkit")
        async def chatkit(request: Request) -> Response:
            return await process_request(
                server, request, DEFAULT_CONTEXT, streaming_response=streaming_response
            )

        with TestClient(app) as client:
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
            assert streamed.headers["cache-control"] == "no-store"
            event = decode_event(streamed.content.splitlines()[0])
            assert isinstance(event, ThreadCreatedEvent)

            loaded = client.post(
                "/chatkit",
                json={
                    "type": "threads.get_by_id",
                    "params": {"thread_id": event.thread.id},
                },
            )
            assert loaded.status_code == 200
            assert "cache-control" not in loaded.headers
            assert loaded.json()["id"] == event.thread.id

            invalid = client.post("/chatkit", content="{")
            assert invalid.status_code == 400
            assert "cache-control" not in invalid.headers


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


@pytest.mark.parametrize("operation", ["items.list", "threads.get_by_id"])
@pytest.mark.parametrize("item_type", [HiddenContextItem, SDKHiddenContextItem])
async def test_history_rejects_a_store_page_containing_hidden_items(
    operation, item_type, monkeypatch: pytest.MonkeyPatch
):
    thread = make_thread()
    item = item_type(
        id="private-item-id",
        thread_id=thread.id,
        created_at=thread.created_at,
        content="private contents",
    )
    visibility_requests = []

    async def broken_page(*args, include_hidden=True, **kwargs):
        visibility_requests.append(include_hidden)
        return Page[ThreadItem](data=[item], has_more=True, after=item.id)

    with (
        make_server() as server,
        make_client(server, raise_server_exceptions=False) as client,
    ):
        await server.store.save_thread(thread, DEFAULT_CONTEXT)
        monkeypatch.setattr(server.store, "load_thread_items", broken_page)
        response = client.post(
            "/chatkit", json={"type": operation, "params": {"thread_id": thread.id}}
        )

    assert visibility_requests == [False]
    assert response.status_code == 500
    assert item.id not in response.text
    assert item.content not in response.text


@pytest.mark.parametrize("state", [TurnState.ACCEPTED, TurnState.RUNNING])
async def test_older_form_conflicts_only_while_newer_reply_is_active(state: TurnState):
    thread = make_thread()
    form = StructuredInputItem(
        id="first-form", thread_id=thread.id, created_at=thread.created_at, inputs=[]
    )
    submission = {
        "type": "threads.add_structured_input",
        "params": {
            "thread_id": thread.id,
            "item_id": form.id,
            "input": {"status": "skipped"},
        },
    }
    ctx = DEFAULT_CONTEXT
    with make_server() as server, make_client(server) as client:
        store = server.store
        await store.begin_turn(thread.id, "first", ctx, create_thread=thread)
        await store.start_turn(thread.id, "first", ctx)
        await store.add_thread_item(thread.id, form, ctx, turn_id="first")
        await store.finish_turn(thread.id, "first", TurnOutcome.WAITING, ctx)
        await store.begin_turn(thread.id, "newer", ctx)
        if state is TurnState.RUNNING:
            await store.start_turn(thread.id, "newer", ctx)

        response = client.post("/chatkit", json=submission)
        assert response.status_code == 409
        assert await store.load_item(thread.id, form.id, ctx) == form
        assert await store.load_turn_state(thread.id, "newer", ctx) is state

        if state is TurnState.ACCEPTED:
            await store.start_turn(thread.id, "newer", ctx)
        await store.finish_turn(thread.id, "newer", TurnOutcome.COMPLETED, ctx)
        assert client.post("/chatkit", json=submission).status_code == 200
        assert await store.load_item(thread.id, form.id, ctx) == form.model_copy(
            update={"status": "skipped"}
        )


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


@pytest.mark.parametrize(
    "result",
    [
        "example",
        {"output": None},
        {"item_id": "tool"},
        {"item_id": "", "output": None},
        {"item_id": "  ", "output": None},
        {"item_id": "tool", "output": None, "unexpected": "value"},
    ],
)
def test_tool_output_requires_a_correlated_envelope(client: TestClient, result: object):
    response = client.post(
        "/chatkit",
        json={
            "type": "threads.add_client_tool_output",
            "params": {"thread_id": "example-thread", "result": result},
        },
    )
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert all(
        issue["location"][:3] == ["threads.add_client_tool_output", "params", "result"]
        for issue in response.json()["detail"]
    )


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
            "params": {
                "thread_id": event.thread.id,
                "result": {"item_id": "unknown-tool", "output": None},
            },
        },
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "ChatKit resource not found."}


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

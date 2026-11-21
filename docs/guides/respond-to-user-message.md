# Respond to a user message

This guide covers how to implement and run a ChatKit server that responds to user messages, including thread loading, inference, event streaming, and persistence.

## Install ChatKit

Follow the [quickstart installation instructions](../quickstart.md#run-your-chatkit-server)
to install the fork and FastAPI for this example.

## Build and run your ChatKit server

The adapter and your ChatKit server work together:

1. Accept HTTP requests from your client.
2. Build a request context in your app, including the authenticated user ID.
3. Dispatch the request and call your `respond` hook for turns that generate a reply.

### Define a request context

First, define a small context object that will be created per request and passed through your server, store, and agents:

```python
from dataclasses import dataclass


@dataclass
class MyRequestContext:
    user_id: str
    staged_response_id: str | None = None
```

### Implement your `ChatKitServer`

Subclass `ChatKitServer` and implement `respond(turn)`. The turn includes
`turn.thread`, `turn.context`, and a typed trigger in `turn.input`.
Yield the events that make up the response.

```python
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from chatkit.errors import InvalidOperationError
from chatkit.events import ThreadItemDoneEvent, ThreadStreamEvent
from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadMetadata,
    UserMessageItem,
)
from chatkit.server import ChatKitServer
from chatkit.turns import Turn


class MyChatKitServer(ChatKitServer[MyRequestContext]):
    async def prepare_user_message(
        self, thread: ThreadMetadata, item: UserMessageItem, context: MyRequestContext
    ) -> UserMessageItem:
        attachment_ids = {attachment.id for attachment in item.attachments}
        if len(attachment_ids) != len(item.attachments):
            raise InvalidOperationError("Attach each file only once.")
        return item

    async def respond(
        self, turn: Turn[MyRequestContext]
    ) -> AsyncIterator[ThreadStreamEvent]:
        # Replace this with your inference pipeline.
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                thread_id=turn.thread.id,
                id=self.store.generate_item_id("message", turn.thread, turn.context),
                created_at=datetime.now(UTC),
                content=[AssistantMessageContent(text="Hi there!")],
            )
        )
```

`Turn.input` distinguishes the four ways ChatKit starts or continues a response.

| Input | Payload |
| --- | --- |
| `UserMessageTurn` | The saved user `.message` on a new or existing thread. |
| `ClientToolTurn` | The saved `.tool_call` with the client's result. |
| `StructuredInputTurn` | The answered or skipped structured input `.item`. |
| `RetryTurn` | The existing user `.message` that starts the retry. |

### Prepare new messages before saving them

Override `prepare_user_message` to check or enrich new user messages. ChatKit
calls it for thread creation and message append after resolving attachments,
before saving the thread, attachment metadata, or message. The thread
snapshot already has an ID, including for a new thread. Return a
`UserMessageItem` with the original ID, thread ID, and creation time. Retries
and continuations use their stored inputs.

Raise `InvalidOperationError` with a client-safe message to return HTTP 422
from the FastAPI adapter before opening an event stream. Raise
`CustomStreamError` with a user-facing message to send an SSE error event.
In either case, ChatKit performs no writes for the rejected message and
reports no turn outcome.

### Observe the turn outcome

Use `on_turn_finished(turn, outcome)` for app work that depends on the
result. Once a turn starts, ChatKit calls the hook after server persistence
or cancellation cleanup. The outcome reports one of four states.

| Outcome | Meaning |
| --- | --- |
| `COMPLETED` | The response finished successfully. |
| `WAITING` | The response left a pending client tool call or structured input. |
| `CANCELLED` | The response was interrupted. |
| `FAILED` | An error stopped the response or was sent in the stream. |

Stage app data in the request context and commit it on the outcome your app
expects. See the examples for [response IDs](#using-previous_response_id-openai-responses-api-only)
and [structured results](#save-structured-tool-results-after-a-turn).
[Turn types](../api/chatkit/turns.md)

An HTTP request can end before the turn starts. Attach request-scoped
resources, such as a lease, to the response lifetime and release them when
the response closes. Track delivery to the browser separately from turn
completion. ChatKit logs exceptions from `on_turn_finished`. If a write
failure needs to reach the browser, make that write in `respond` before
completing the turn.

### Wire ChatKit to your web framework

Expose a single `/chatkit` endpoint. Install the `fastapi` extra to use the
adapter with your `MyChatKitServer` instance. Construct the request context
in your app and pass it into the adapter.

```python
from fastapi import FastAPI, Request, Response

from chatkit.fastapi import process_request

app = FastAPI()
store = MyPostgresStore(conn_info)
server = MyChatKitServer(store)


@app.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    # Build a per-request context from the incoming HTTP request.
    context = MyRequestContext(user_id="abc123")

    return await process_request(server, request, context)
```

The adapter serves JSON or an SSE stream. It returns HTTP 400 for malformed
JSON, HTTP 422 for an invalid or unsupported operation, HTTP 404 for a
missing resource, and HTTP 409 for a conflict with the current thread state.
Once streaming starts, ChatKit sends error events in the SSE response.
[FastAPI adapter reference](../api/chatkit/fastapi.md)

For custom buffering or send deadlines, pass
`streaming_response=your_response_factory`. ChatKit validates and prepares
the typed request before calling the factory with a `StreamingResult`,
including committing any pending-item claim or retry truncation.
Immediate results and HTTP errors use the adapter's regular responses.

Use `ChatKitStreamingResponse(result)` for the SDK's framing and disconnect
handling. If you supply your own streaming response, it owns the result.
Use `encode_sse` for each event and await `result.aclose()` when the response
ends, fails, or disconnects. Stop and await any active reader before closing
the result. Protect cleanup from cancellation so pending store writes can
finish.

### How request context flows into ChatKit

`ChatKitServer[TContext]` and the storage interfaces use a request context type
you choose. Construct it per request and pass it to `server.process`. Read it
from `turn.context` in the turn hooks and from the `context` argument in your
store methods.

```python
from chatkit.protocol import parse_request

context = MyRequestContext(user_id="abc123")
result = await server.process(parse_request(await request.body()), context)
```

Build authorization fields in the context from your authenticated application
session. Treat metadata in the request body as client input.
With FastAPI, pass the server, the HTTP request, and your context to
`chatkit.fastapi.process_request`. For custom transports, use `parse_request`
on inbound JSON and `encode_message` or `encode_sse` for outbound JSON or
SSE. The encoders wrap pending browser tool arguments as `{item_id, input}`.
Application events, models, and storage keep the browser arguments and
output unwrapped. A custom encoder must preserve the wire format in both streamed
events and saved items loaded by the browser. The inbound operation supplies
the browser's `{item_id, output}` inside `params.result`. Keep
`ClientToolCallItem.model_continuation` server-side when encoding a browser
response.
[Protocol helpers](../api/chatkit/protocol.md) ·
[Browser callback contract](update-client-during-response.md#register-the-client-tool-in-chatkitjs)

## Implement your ChatKit data store

Implement `ThreadStore` to persist threads, messages, tool calls, and widgets.
The [in-memory example](../examples/in_memory_thread_store.py) implements all
thread and item methods. Implement `claim_pending_item`, `patch_thread`, and
`truncate_after` as short atomic operations in your database. Use
`patch_thread` to edit saved fields, and yield the result of
`ChatKitServer.update_thread` when a connected client should see an edit.
For `load_thread_items(..., include_hidden=False)`, filter the model-only items
before computing the page and its cursor.

For a retry, `truncate_after` returns the retained `UserMessageItem`. Before
returning, atomically make later items disappear from pagination and direct
`load_item` reads using the same stable order as `load_thread_items`. For
stores with a bounded transaction size, commit the visibility change in one
operation and reclaim inactive records separately under your retention
policy. Test retrying a history larger than a single write batch.

To support uploads, supply an
`AttachmentMetadataStore` for records and an `AttachmentStore` for file
operations. [Attachment setup](accept-rich-user-input.md#configure-an-upload-strategy)

### Implement the client tool lifecycle

Implement `load_client_tool_status(thread_id, item_id, context)` and
`cancel_client_tool_call(thread_id, item_id, context)` when your app uses
browser tools. The state belongs to the same authenticated owner, thread, and
item as the tool call. Keep it durable and independent of the transcript's
ordering or visibility. Maintain at most one pending call per scoped thread.

- In `add_thread_item`, create a pending `ClientToolCallItem` and record
  its pending status in one transaction. Reject a second pending call or
  reuse of any recorded callback ID in that thread. Inserting a new user
  message cancels the pending call in the same transaction, including when
  `save_item` inserts the message. Use `claim_pending_item` to complete a
  tool call and reject direct tool item writes through `save_item`.
- Store `model_continuation` with the call and preserve it when completing
  the call. It can contain native model inputs and outputs. Apply your
  application's access and retention rules for model history to that data.
- In `claim_pending_item`, compare the exact saved pending item and its
  pending status, then save the completed item and its completed status in
  the same transaction. Only one claimant can succeed. If you use
  `validate_pending_claim`, call it with the existing item, expected item,
  and replacement. The `latest_item_id` argument was removed.
- In `truncate_after`, cancel the pending call as part of the retry.
  In `delete_thread_item`, cancel the pending call if its item is deleted.
  Keep lifecycle records after retry or item deletion so
  `load_client_tool_status` can still report their state. In
  `delete_thread`, remove those records with the rest of the thread.
- In browser-visible pagination (`include_hidden=False`), exclude pending
  tool items with cancelled or unregistered callbacks before calculating
  the page and cursor.
- Make `cancel_client_tool_call` safe to repeat. It cancels only the
  specified pending ID. If the call is already completed or cancelled, or a
  different ID is pending, keep that state unchanged. ChatKit uses this
  during cleanup when a stream fails to finish its browser-tool handoff.

Store implementations can use the [in-memory](../examples/in_memory_thread_store.py)
and [SQLite](../examples/sqlite_thread_store.py) examples as references.
Test concurrent claims and verify that cancellation and completion remain
effective after reopening durable storage. Treat the returned tool output
as client input before using it in any application decision.

### Use the file-backed SQLite example

The [SQLite example](../examples/sqlite_thread_store.py) implements
`ThreadStore` with the client tool lifecycle. Save it with your app or import
it from this checkout.
It scopes thread and item queries by `context.user_id`, so supply this value
from your authenticated application session. The database survives restarts.
Use a new SQLite file for the example.

```python
from dataclasses import dataclass

from docs.examples.sqlite_thread_store import SQLiteThreadStore


@dataclass(frozen=True)
class RequestContext:
    user_id: str


store = SQLiteThreadStore[RequestContext]("chat.sqlite3")
```

Pass `store` to your `ChatKitServer` subclass. The example uses short SQLite
transactions and Python's synchronous `sqlite3` driver. Each operation runs
on the calling thread, so this is a local development reference. For a busy
async server, use an async driver or run the SQLite transactions in a worker.
The example persists threads and items. Configure a separate attachment store
if your app supports uploads.

### Keep old and new timestamps sortable

Turnspan writes timestamps with a UTC offset. Existing records may contain
timestamps without an offset. Find the timezone used by the old writer and
use it to interpret those records. For example, if the old writer ran in
UTC, pass `UTC` as `legacy_timezone`.

```python
from datetime import UTC, datetime, tzinfo


def normalize_created_at(value: datetime, legacy_timezone: tzinfo) -> datetime:
    if value.utcoffset() is None:
        value = value.replace(tzinfo=legacy_timezone)
    return value.astimezone(UTC)
```

Use the same `(normalized_created_at, id)` ordering for history pages,
cursor lookups, and retry truncation. If your database stores a separate
sort key, apply the same normalization to its old and new keys. Check
pages and retries that cross the boundary between legacy and newer items.

### Check your app's store

Run `check_thread_store` against a store that supports browser tools using
a disposable database and contexts for two distinct test users. The check
writes and deletes a synthetic thread. It verifies scoped reads,
tied-timestamp pagination, concurrent patches, form claims, callback
completion and cancellation, and retry truncation.

```python
import pytest

from chatkit.store_checks import check_thread_store


@pytest.mark.asyncio
async def test_app_store(tmp_path):
    path = tmp_path / "contract.sqlite3"
    await check_thread_store(
        SQLiteThreadStore[RequestContext](path),
        RequestContext(user_id="example-user"),
        other_context=RequestContext(user_id="other-example-user"),
        peer=SQLiteThreadStore[RequestContext](path),
    )
```

For async adapters, supply independent store instances sharing the test database
so the two calls can contend across connections. A synchronous driver runs
those tasks in sequence on an event loop. To test actual write contention
with synchronous SQLite, run each transaction on its own worker, as in the
repository's SQLite concurrency test.

Customize ID generation by overriding `generate_thread_id` and `generate_item_id` if you need external or deterministic IDs. Store metadata such as a model `last_response_id` on `ThreadMetadata` to drive your inference pipeline.

## Generate a response using your model

Inside `respond`, you'll usually:

1. Load recent thread history.
2. Prepare model input for your agent.
3. Run inference and stream events back to the client.

### Prepare model input

Before you call your model, decide what conversation context you want the model to see.

#### Load thread history (recommended default)

Fetch recent items to give the model descriptive context from the thread.
The default converter omits client tool items. Native Responses replay
requires the provider's original items, including reasoning and assistant
phase when supplied. For an accepted browser result, use the saved
[client tool continuation](update-client-during-response.md#build-the-next-model-request).
[OpenAI reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning#keeping-reasoning-items-in-context)

```python
items_page = await self.store.load_thread_items(
    turn.thread.id,
    after=None,
    limit=20,  # Tune this limit based on your model/context budget.
    order="desc",
    context=turn.context,
)
items = list(reversed(items_page.data))
```

- Start with the defaults: `simple_to_agent_input(items)` is a convenience wrapper around the **default** `ThreadItemConverter` (it calls `ThreadItemConverter().to_agent_input(items)` under the hood).
- Customize for your integration: some item types require app-specific translation into model input (for example: **attachments**, **tags**, and some **hidden context items**). In those cases, subclass `ThreadItemConverter` and call your converter directly instead of `simple_to_agent_input`. (If your thread includes attachments/tags and you haven't implemented the converter hooks, conversion will raise `NotImplementedError`.)

```python
from chatkit.agent_input import simple_to_agent_input
from chatkit.agents import AgentContext


# Option A (defaults):
input_items = await simple_to_agent_input(items)

# Option B (your integration-specific converter):
# input_items = await MyThreadItemConverter().to_agent_input(items)

agent_context = AgentContext(
    thread=turn.thread,
    store=self.store,
    request_context=turn.context,
    resume_from=None,
)
```

Read client model and tool choices from `turn.input.message.inference_options`
on a `UserMessageTurn` or `RetryTurn`. Validate the choices against the
models and tools your app offers.

#### Using `previous_response_id` (OpenAI Responses API only)

The OpenAI Responses API can retrieve model-visible history from a stored
response. If your app permits provider-side storage, pass a saved response ID
as `previous_response_id` to `Runner.run_streamed()` on the next turn. You
can build descriptive context for ChatKit-only items such as widgets using
the [thread history approach above](#load-thread-history-recommended-default).
For local native replay, preserve the original model input and output items
in order. [OpenAI conversation state](https://developers.openai.com/api/docs/guides/conversation-state)

This responder handles user messages and retries. It enables provider-side
storage for the run. ChatKit saves the incoming user message before calling
`respond`, so the descriptive history fallback already includes it. If your
app also uses browser tools, handle `ClientToolTurn` with the saved
[client tool continuation](update-client-during-response.md#build-the-next-model-request).

```python
from contextlib import aclosing

from agents import ModelSettings, RunConfig, Runner

from chatkit.agent_input import simple_to_agent_input
from chatkit.agents import AgentContext, stream_agent_response
from chatkit.turns import Turn, TurnOutcome, UserMessageTurn


class MyChatKitServer(ChatKitServer[MyRequestContext]):
    async def respond(
        self, turn: Turn[MyRequestContext]
    ) -> AsyncIterator[ThreadStreamEvent]:
        thread, context = turn.thread, turn.context
        saved_id = thread.metadata.get("last_response_id")
        if saved_id is not None:
            await self.store.patch_thread(
                thread.id, {"metadata_delete": ("last_response_id",)}, context
            )

        previous_response_id = None
        if (
            isinstance(saved_id, str)
            and saved_id
            and isinstance(turn.input, UserMessageTurn)
        ):
            previous_response_id = saved_id
            input_items = await simple_to_agent_input(turn.input.message)
        else:
            items_page = await self.store.load_thread_items(
                thread.id, after=None, limit=20, order="desc", context=context
            )
            input_items = await simple_to_agent_input(list(reversed(items_page.data)))

        agent_context = AgentContext(
            thread=thread, store=self.store, request_context=context, resume_from=None
        )
        result = Runner.run_streamed(
            assistant_agent,
            input_items,
            context=agent_context,
            previous_response_id=previous_response_id,
            run_config=RunConfig(model_settings=ModelSettings(store=True)),
        )

        async with aclosing(stream_agent_response(agent_context, result)) as events:
            async for event in events:
                yield event
        context.staged_response_id = result.last_response_id

    async def on_turn_finished(
        self, turn: Turn[MyRequestContext], outcome: TurnOutcome
    ) -> None:
        if outcome is TurnOutcome.COMPLETED and turn.context.staged_response_id:
            await self.store.patch_thread(
                turn.thread.id,
                {"metadata_set": {"last_response_id": turn.context.staged_response_id}},
                turn.context,
            )
```

`result.last_response_id` becomes available as the runner receives model
responses. Save it only after the stream finishes successfully. If the provider
reports that the saved response is no longer retrievable, a fresh user-message
run can use descriptive thread context or the original input and output
history available to your app. Resuming a browser tool requires its matching
continuation. Only retry automatically if nothing has been sent to the client
and no tools have taken action.

### Prepare workflow progress

Choose the workflow state before `Runner.run_streamed` schedules the agent
run. For a fresh run, pass `resume_from=None` when building the context, as
in the examples above. Tools can still start, update, and finish progress
during that run. The stream helper does no history lookup for workflow
discovery.

To continue saved progress, use your server's continuation state to verify
that the operation can still resume and obtain its exact workflow item ID.
The following example assumes your application already authorized and
verified that association. It loads the item through the thread store
before starting the agent.

```python
from agents import Runner

from chatkit.agents import AgentContext
from chatkit.errors import InvalidOperationError
from chatkit.models import WorkflowItem

saved_workflow = await self.store.load_item(
    turn.thread.id, workflow_item_id, context=turn.context
)
if not isinstance(saved_workflow, WorkflowItem):
    raise InvalidOperationError("The saved item is not a workflow.")

agent_context = AgentContext(
    thread=turn.thread,
    store=self.store,
    request_context=turn.context,
    resume_from=saved_workflow,
)
result = Runner.run_streamed(assistant_agent, input_items, context=agent_context)
```

You can also pass a verified `WorkflowItem` your server has already loaded.
Keep the association in durable application storage if it needs to survive a
restart. Handle missing or stale continuations before starting the agent.
When the run updates the resumed workflow, the client receives changes under
the original item ID.

This choice controls displayed workflow progress. Prepare model input
independently using the history or provider continuation approach you chose.

### Run inference and stream events

Run your agent and stream events back to the client. `stream_agent_response` converts an Agents run into ChatKit events; you can also yield events manually.

```python
from contextlib import aclosing

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
)
from chatkit.agents import stream_agent_response
from chatkit.events import ErrorEvent

result = Runner.run_streamed(
    assistant_agent,
    input_items,
    context=agent_context,
)

try:
    async with aclosing(stream_agent_response(agent_context, result)) as events:
        async for event in events:
            yield event
except InputGuardrailTripwireTriggered:
    yield ErrorEvent(message="We blocked that message for safety.")
except OutputGuardrailTripwireTriggered:
    yield ErrorEvent(
        message="The assistant response was blocked.",
        allow_retry=False,
    )
```

To stream events from a server tool during the same turn, use `ctx.context.stream(...)` inside the tool:

```python
from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.events import ProgressUpdateEvent


@function_tool()
async def load_document(ctx: RunContextWrapper[AgentContext], document_id: str):
    await ctx.context.stream(
        ProgressUpdateEvent(icon="document", text="Loading document...")
    )
    return await get_document_by_id(document_id)
```

`stream_agent_response` forwards these events alongside assistant text and tool updates. To call a browser callback from an agent tool, use [`request_client_tool`](update-client-during-response.md#define-a-client-tool-in-your-agent).

### Save structured tool results after a turn

Use the same `AgentContext` for the run and the ChatKit stream. Inside a tool,
`ctx.context.thread.id` gives you the current ChatKit thread ID. Extend the
context to hold structured output until the turn finishes. The example below
keeps one feedback record per user message.

```python
from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel

from chatkit.agents import AgentContext


class Feedback(BaseModel):
    summary: str


@dataclass
class FeedbackRequestContext(MyRequestContext):
    feedback: Feedback | None = None


class FeedbackContext(AgentContext[FeedbackRequestContext]):
    feedback: Feedback | None = None


formatter = Agent[FeedbackContext](
    name="Feedback formatter",
    instructions="Summarize the user's feedback in one sentence.",
    output_type=Feedback,
)


@function_tool(failure_error_function=None)
async def collect_feedback(ctx: RunContextWrapper[FeedbackContext], text: str) -> str:
    """Collect feedback the user shared in this turn."""
    formatted = await Runner.run(formatter, text, context=ctx.context)
    ctx.context.feedback = formatted.final_output_as(
        Feedback, raise_if_incorrect_type=True
    )
    return "Feedback collected."


feedback_agent = Agent[FeedbackContext](
    name="Assistant",
    instructions="Call collect_feedback when the user shares feedback.",
    tools=[collect_feedback],
)
```

Pass a `FeedbackRequestContext` to `server.process`. Supply
`self.feedback_repository` from your application with an async `upsert`
method. Use a unique key of user ID, thread ID, and source message ID so
retrying the same message updates its existing record.

```python
from contextlib import aclosing

from chatkit.agent_input import simple_to_agent_input
from chatkit.agents import stream_agent_response
from chatkit.turns import RetryTurn, Turn, TurnOutcome, UserMessageTurn


class FeedbackChatKitServer(ChatKitServer[FeedbackRequestContext]):
    async def respond(
        self, turn: Turn[FeedbackRequestContext]
    ) -> AsyncIterator[ThreadStreamEvent]:
        thread, context = turn.thread, turn.context
        items_page = await self.store.load_thread_items(
            thread.id, after=None, limit=20, order="desc", context=context
        )
        input_items = await simple_to_agent_input(list(reversed(items_page.data)))
        agent_context = FeedbackContext(
            thread=thread, store=self.store, request_context=context, resume_from=None
        )
        result = Runner.run_streamed(feedback_agent, input_items, context=agent_context)

        async with aclosing(stream_agent_response(agent_context, result)) as events:
            async for event in events:
                yield event

        if (
            result.is_complete
            and not result.interruptions
            and result.final_output is not None
        ):
            context.feedback = agent_context.feedback

    async def on_turn_finished(
        self, turn: Turn[FeedbackRequestContext], outcome: TurnOutcome
    ) -> None:
        if (
            outcome is TurnOutcome.COMPLETED
            and isinstance(turn.input, (UserMessageTurn, RetryTurn))
            and turn.context.feedback is not None
        ):
            await self.feedback_repository.upsert(
                user_id=turn.context.user_id,
                thread_id=turn.thread.id,
                source_message_id=turn.input.message.id,
                feedback=turn.context.feedback,
            )
```

The completion hook writes feedback only for a successful turn with a source
user message. ChatKit reports cancellations, guardrail errors, and turns
waiting for client input separately.

This example builds on Jao Garcia's persistence question. [Upstream issue #120](https://github.com/openai/chatkit-python/issues/120)

## Next: add features

- [Let users browse past threads](browse-past-threads.md)
- [Accept rich user input](accept-rich-user-input.md)
- [Let users pick tools and models](let-users-pick-tools-and-models.md)
- [Pass extra app context to your model](pass-extra-app-context-to-your-model.md)
- [Update the client during a response](update-client-during-response.md)
- [Build interactive responses with widgets](build-interactive-responses-with-widgets.md)
- [Add annotations in assistant messages](add-annotations.md)
- [Stream generated images](stream-generated-images.md)
- [Keep your app in sync with ChatKit](keep-your-app-in-sync-with-chatkit.md)
- [Let your app draft and send messages](let-your-app-draft-and-send-messages.md)
- [Handle feedback](handle-feedback.md)
- [Prepare your app for production](prepare-your-app-for-production.md)

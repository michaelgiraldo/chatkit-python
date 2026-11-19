# Respond to a user message

This guide covers how to implement and run a ChatKit server that responds to user messages, including thread loading, inference, event streaming, and persistence.

## Install ChatKit

Follow the [quickstart installation instructions](../quickstart.md#run-your-chatkit-server)
to install the fork and FastAPI for this example.

## Build and run your ChatKit server

Your ChatKit server does three main things:

1. Accept HTTP requests from your client.
2. Construct a request context (user id, auth, feature flags, etc.).
3. Call `ChatKitServer.respond` to produce streamed events.

### Define a request context

First, define a small context object that will be created per request and passed through your server, store, and agents:

```python
from dataclasses import dataclass


@dataclass
class MyRequestContext:
    user_id: str
```

### Implement your `ChatKitServer`

Subclass `ChatKitServer` and implement `respond`. It runs once per user turn and should yield the events that make up your response. We'll keep this example simple for now and fill in history loading and model calls in later sections.

```python
from collections.abc import AsyncIterator
from datetime import datetime

from chatkit.server import ChatKitServer
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)


class MyChatKitServer(ChatKitServer[MyRequestContext]):
    async def respond(
        self,
        thread: ThreadMetadata,
        input: UserMessageItem | None,
        context: MyRequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        # Replace this with your inference pipeline.
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                thread_id=thread.id,
                id=self.store.generate_item_id("message", thread, context),
                created_at=datetime.now(),
                content=[AssistantMessageContent(text="Hi there!")],
            )
        )
```

### Wire ChatKit to your web framework

Expose a single `/chatkit` endpoint that forwards requests to your
`MyChatKitServer` instance. For this FastAPI example, save
[chatkit_response.py](../examples/chatkit_response.py) alongside your endpoint module.

```python
from fastapi import FastAPI, Request, Response

from chatkit.server import StreamingResult
from chatkit_response import ChatKitStreamingResponse

app = FastAPI()
store = MyPostgresStore(conn_info)
server = MyChatKitServer(store)


@app.post("/chatkit")
async def chatkit_endpoint(request: Request):
    # Build a per-request context from the incoming HTTP request.
    context = MyRequestContext(user_id="abc123")

    # Let ChatKit handle the request and return either a streaming or JSON result.
    result = await server.process(await request.body(), context)
    if isinstance(result, StreamingResult):
        return ChatKitStreamingResponse(result)
    return Response(content=result.json, media_type="application/json")
```

### How request context flows into ChatKit

`ChatKitServer[TContext]` and `Store[TContext]` are generic over a request context type you choose (user id, org, auth scopes, feature flags). Construct it per request and pass it to `server.process`; it flows into `respond` and your store methods.

```python
context = MyRequestContext(user_id="abc123")
result = await server.process(await request.body(), context)
```

Request metadata in the payload is available before calling `process`; include it in your context for auth, tracing, or feature flags.

## Implement your ChatKit data store

Implement the `Store` interface to control how threads, messages, tool calls, and widgets are stored. Prefer serializing thread items as JSON so schema changes do not break storage. Example Postgres store:

```python
class MyPostgresStore(Store[RequestContext]):
    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    data JSONB NOT NULL
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL
                        REFERENCES threads (id)
                        ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    data JSONB NOT NULL
                );
                """
            )

            conn.commit()

    async def load_thread(
        self, thread_id: str, context: RequestContext
    ) -> ThreadMetadata:
        with self._connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
            cur.execute(
                "SELECT data FROM threads WHERE id = %s AND user_id = %s",
                (thread_id, context.user_id),
            )
            row = cur.fetchone()
            if row is None:
                raise NotFoundError(f"Thread {thread_id} not found")
            return ThreadMetadata.model_validate(row[0])

    async def save_thread(
        self, thread: ThreadMetadata, context: RequestContext
    ) -> None:
        payload = thread.model_dump(mode="json")
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO threads (id, user_id, created_at, data)
                VALUES (%s, %s, %s, %s)
                """,
                (thread.id, context.user_id, thread.created_at, payload),
            )
            conn.commit()

    # Implement the remaining Store methods following the same pattern.
```

Customize ID generation by overriding `generate_thread_id` and `generate_item_id` if you need external or deterministic IDs. Store metadata such as a model `last_response_id` on `ThreadMetadata` to drive your inference pipeline.

## Generate a response using your model

Inside `respond`, you'll usually:

1. Load recent thread history.
2. Prepare model input for your agent.
3. Run inference and stream events back to the client.

### Prepare model input

Before you call your model, decide what conversation context you want the model to see.

#### Load thread history (recommended default)

Fetch recent items so the model sees the conversation state before you build the next turn:

```python
items_page = await self.store.load_thread_items(
    thread.id,
    after=None,
    limit=20,  # Tune this limit based on your model/context budget.
    order="desc",
    context=context,
)
items = list(reversed(items_page.data))
```

- Start with the defaults: `simple_to_agent_input(items)` is a convenience wrapper around the **default** `ThreadItemConverter` (it calls `ThreadItemConverter().to_agent_input(items)` under the hood).
- Customize for your integration: some item types require app-specific translation into model input (for example: **attachments**, **tags**, and some **hidden context items**). In those cases, subclass `ThreadItemConverter` and call your converter directly instead of `simple_to_agent_input`. (If your thread includes attachments/tags and you haven't implemented the converter hooks, conversion will raise `NotImplementedError`.)

```python
from agents import Runner

from chatkit.agents import AgentContext, ThreadItemConverter, simple_to_agent_input


# Option A (defaults):
input_items = await simple_to_agent_input(items)

# Option B (your integration-specific converter):
# input_items = await MyThreadItemConverter().to_agent_input(items)

agent_context = AgentContext(
    thread=thread,
    store=self.store,
    request_context=context,
)
```

Respect any `input.inference_options` the client sends (model, tool choice, etc.) when you build your request to the model.

#### Using `previous_response_id` (OpenAI Responses API only)

The OpenAI Responses API can retrieve model-visible history from a stored
response. If your app permits provider-side storage, pass a saved response ID
as `previous_response_id` to `Runner.run_streamed()` on the next turn. For
locally managed history or ChatKit-only items such as widgets, use the
[thread history approach above](#load-thread-history-recommended-default).

Use this example inside `respond` in place of the history-only setup and
streaming example. It enables provider-side storage for the run. ChatKit
saves the incoming user message before calling `respond`, so the history
fallback already includes that message.

```python
from contextlib import aclosing

from agents import ModelSettings, RunConfig, Runner

from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.types import ErrorEvent

agent_context = AgentContext(thread=thread, store=self.store, request_context=context)
saved_id = thread.metadata.pop("last_response_id", None)
if saved_id is not None:
    # An interrupted turn leaves the next turn free to rebuild from thread history.
    await self.store.save_thread(thread, context=context)

if isinstance(saved_id, str) and saved_id and input is not None:
    previous_response_id = saved_id
    input_items = await simple_to_agent_input([input])
else:
    previous_response_id = None
    items_page = await self.store.load_thread_items(
        thread.id, after=None, limit=20, order="desc", context=context
    )
    input_items = await simple_to_agent_input(list(reversed(items_page.data)))

result = Runner.run_streamed(
    assistant_agent,
    input_items,
    context=agent_context,
    previous_response_id=previous_response_id,
    run_config=RunConfig(model_settings=ModelSettings(store=True)),
)

stream_failed = False
async with aclosing(stream_agent_response(agent_context, result)) as events:
    async for event in events:
        if isinstance(event, ErrorEvent):
            stream_failed = True
        yield event

if not stream_failed and result.last_response_id:
    thread.metadata["last_response_id"] = result.last_response_id
    await self.store.save_thread(thread, context=context)
```

`result.last_response_id` becomes available as the runner receives model
responses. Save it only after the stream finishes successfully. If the provider
reports that the saved response is no longer retrievable, a fresh attempt can
rebuild its input from thread history. Only retry automatically if nothing has
been sent to the client and no tools have taken action.

### Run inference and stream events

Run your agent and stream events back to the client. `stream_agent_response` converts an Agents run into ChatKit events; you can also yield events manually.

```python
from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
)
from chatkit.agents import stream_agent_response
from chatkit.types import ErrorEvent

result = Runner.run_streamed(
    assistant_agent,
    input_items,
    context=agent_context,
)

try:
    async for event in stream_agent_response(agent_context, result):
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
from chatkit.types import ProgressUpdateEvent


@function_tool()
async def load_document(ctx: RunContextWrapper[AgentContext], document_id: str):
    await ctx.context.stream(
        ProgressUpdateEvent(icon="document", text="Loading document...")
    )
    return await get_document_by_id(document_id)
```

`stream_agent_response` will forward these events alongside any assistant text or tool call updates. Client tool calls are also supported via `ctx.context.client_tool_call` when you register the tool on both client and server.

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


class FeedbackContext(AgentContext[MyRequestContext]):
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

In `respond`, build `input_items` from thread history as above and use this
streaming block. Supply `self.feedback_repository` from your application with
an async `upsert` method. Use a unique key of user ID, thread ID, and source
message ID so retrying the same message updates its existing record.

```python
from contextlib import aclosing

from chatkit.agents import stream_agent_response
from chatkit.types import ErrorEvent

agent_context = FeedbackContext(
    thread=thread, store=self.store, request_context=context
)
result = Runner.run_streamed(feedback_agent, input_items, context=agent_context)

stream_failed = False
async with aclosing(stream_agent_response(agent_context, result)) as events:
    async for event in events:
        if isinstance(event, ErrorEvent):
            stream_failed = True
        yield event

feedback = agent_context.feedback
if (
    not stream_failed
    and result.is_complete
    and not result.interruptions
    and result.final_output is not None
    and input is not None
    and feedback is not None
):
    await self.feedback_repository.upsert(
        user_id=context.user_id,
        thread_id=thread.id,
        source_message_id=input.id,
        feedback=feedback,
    )
```

Keep the upsert after the stream so exceptions, guardrail blocks, and early
stream closure leave the staged result unwritten. If you use the guardrail
handlers above, return after yielding the error event.
Runs without a source user message skip this example's write.

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

# Update the client during a response

Keep your UI responsive while the server is working: stream progress text, trigger client-side effects, and run client tools mid-response without blocking everything else.

This guide covers three patterns:

- Progress updates for lightweight status while tools run
- Client effects for fire-and-forget UI behavior
- Client tools for round-trips to the browser/app during inference

## Show progress while tools run

Use `ProgressUpdateEvent` when you need lightweight, real-time status. These updates stream immediately to the client and disappear after the turn—they are not stored in the thread.

### From tools

Inside a tool, use `AgentContext.stream` to enqueue progress events. They are delivered to the client immediately and are not persisted as thread items.

```python
from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.events import ProgressUpdateEvent


@function_tool()
async def ingest_files(ctx: RunContextWrapper[AgentContext], paths: list[str]):
    await ctx.context.stream(ProgressUpdateEvent(icon="document", text="Uploading..."))
    await upload(paths)

    await ctx.context.stream(
        ProgressUpdateEvent(icon="search", text="Indexing and chunking...")
    )
    await index_files(paths)

    await ctx.context.stream(ProgressUpdateEvent(icon="check", text="Done"))
```

`stream_agent_response` will forward these events for you alongside any assistant text or tool call updates.

### From custom pipelines

Custom pipelines can yield `ProgressUpdateEvent` directly from `respond` or
`action`. This example uses your app's `RequestContext`, `search_tickets`, and
`stream_summary` helpers. `stream_summary` yields ChatKit events.

```python
from collections.abc import AsyncIterator

from chatkit.turns import Turn
from chatkit.events import ProgressUpdateEvent, ThreadStreamEvent


async def respond(self, turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
    yield ProgressUpdateEvent(icon="search", text="Searching tickets...")
    results = await search_tickets(turn.context)

    yield ProgressUpdateEvent(icon="square-code", text="Generating summary...")
    async for event in stream_summary(results):
        yield event
```

Use short, action-oriented messages and throttle updates to meaningful stages instead of every percent to avoid noisy streams.

## Trigger client-side effects without blocking

Send `ClientEffectEvent` to trigger fire-and-forget UI work (such as refreshing a view, opening a modal, or showing a toast) without creating thread items or pausing the model stream.

Client effects are ephemeral: they stream immediately to ChatKit.js, trigger your registered effect handler, and are not persisted to the thread history. Use client tool calls instead when you need a round-trip response from the client.

### Stream a client effect from your server

Yield client effects directly from the `respond` or `action` method:

```python
from collections.abc import AsyncIterator

from chatkit.turns import Turn
from chatkit.events import ClientEffectEvent, ThreadStreamEvent


async def respond(self, turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
    yield ClientEffectEvent(
        name="highlight_text",
        data={"index": 142, "length": 35},
    )
```

Or from tools, through `AgentContext`:

```python
from agents import RunContextWrapper, function_tool
from chatkit.agents import AgentContext
from chatkit.events import ClientEffectEvent


@function_tool()
async def highlight_text(ctx: RunContextWrapper[AgentContext], index: int, length: int):
    await ctx.context.stream(
        ClientEffectEvent(
            name="highlight_text",
            data={"index": index, "length": length},
        )
    )
```

### Handle the client effect in ChatKit.js

Register a client effect handler when initializing ChatKit on the client:

```ts
const chatkit = useChatKit({
  // ...
  onEffect: async ({name, data}) => {
    if (
      name === "highlight_text" &&
      typeof data?.index === "number" &&
      typeof data.length === "number"
    ) {
      highlightArticleText({index: data.index, length: data.length});
    }
  },
});
```

## Call client tools mid-inference

Client tool calls let the agent invoke browser or app callbacks
mid-inference. Register the tool on the client and server. When the model
calls the tool, ChatKit pauses the model, sends the tool request to the
client, and resumes with the returned result.

Use client effects instead when you do not need to wait for the client callback response for the rest of your response.

### Define a client tool in your agent

Annotate the tool with the Agents SDK's `ToolContext` and call `ctx.context.request_client_tool(ctx, ...)` to tie the browser callback to this invocation. Turnspan accepts one client tool request per turn. Include the client tool in `stop_at_tool_names` so the model pauses while the browser callback runs and returns its result.

```python
from agents import Agent, StopAtTools, function_tool
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext


@function_tool(description_override="Read the user's current canvas selection.")
async def get_selected_canvas_nodes(ctx: ToolContext[AgentContext]) -> None:
    ctx.context.request_client_tool(
        ctx,
        name="get_selected_canvas_nodes",
        arguments={"project": my_project()},
    )


assistant = Agent[AgentContext](
    ...,
    tools=[get_selected_canvas_nodes],
    # StopAtTools pauses model generation so the pending client callback can run and resume the run.
    tool_use_behavior=StopAtTools(stop_at_tool_names=[get_selected_canvas_nodes.name]),
)
```

### Register the client tool in ChatKit.js

Register an `onClientTool` callback that handles the name passed to
`request_client_tool`. Turnspan supplies the saved item ID and the tool
arguments as `params.item_id` and `params.input`. Return the same item ID
with your tool's JSON output. The canvas tool below takes a project ID
under `input.project` and returns the selected nodes under `output.nodes`.
[ChatKit.js callback API](https://openai.github.io/chatkit-js/api/openai/chatkit/type-aliases/chatkitoptions/#onclienttool)

```ts
const chatkit = useChatKit({
  // ...
  onClientTool: async ({name, params}) => {
    const {item_id, input} = params;
    if (
      name !== "get_selected_canvas_nodes" ||
      typeof item_id !== "string" ||
      !item_id ||
      typeof input !== "object" ||
      input === null ||
      !("project" in input) ||
      typeof input.project !== "string"
    ) {
      throw new Error("Unexpected client tool request");
    }

    const nodes = myCanvas.getSelectedNodes(input.project);
    return {
      item_id,
      output: {nodes: nodes.map((node) => ({id: node.id, kind: node.type}))},
    };
  },
});
```

### Stream and resume

In `respond(self, turn)`, use `turn.thread` and `turn.context` to construct
your `AgentContext`, then stream via `stream_agent_response`. ChatKit saves a
pending client tool call before emitting the request to the browser. The
agent helper emits a `ThreadItemDoneEvent` for the pending tool item. Use that
event if you build the tool request directly in a custom responder.

ChatKit.js posts the callback's return value as `params.result` in the
existing `threads.add_client_tool_output` operation.

```json
{
  "type": "threads.add_client_tool_output",
  "params": {
    "thread_id": "thr_example",
    "result": {
      "item_id": "tc_example",
      "output": {"nodes": [{"id": "node_1", "kind": "text"}]}
    }
  }
}
```

The Python request model exposes that result as
[`ClientToolCallbackResult`](../api/chatkit/requests.md#chatkit.requests.ClientToolCallbackResult).
The server uses the authenticated request context, the thread ID, and the
item ID to find the pending invocation. It commits the claim and output before
calling `respond` again. On that turn, `turn.input` is a `ClientToolTurn` and
`turn.input.tool_call.output` contains the inner `output` value. The
`ThreadItemConverter.client_tool_call_to_input` method supplies the model with
the original tool arguments, the output, and the provider's original
`call_id`. The callback envelope belongs to the browser transport.

For each run, pass `resume_from=None` to start fresh or prepare an explicitly
linked [saved workflow](respond-to-user-message.md#prepare-workflow-progress)
before starting the agent. Use the claimed tool item to resolve a linked
workflow from trusted storage if your app needs to resume one.

### Know when a callback can complete

Each thread can have one pending client tool call. A completed handoff leaves
that exact call pending so a callback can arrive later, even after the server
restarts with a durable store. New messages and retries cancel the pending
call. Deleting the call cancels it, and deleting the thread invalidates every
callback for that thread. Closing or failing a stream before it finishes the
handoff cancels the exact call the stream opened if it is still pending.
A result that already won the atomic claim stays completed.

A callback for a missing, already completed, cancelled, or inaccessible
invocation cannot resume the turn. The server matches the saved ID even when
other items were written to the transcript after the tool request. See
[implement the client tool lifecycle](respond-to-user-message.md#implement-the-client-tool-lifecycle)
for storage requirements.

The lifecycle governs calls that a responder has already registered. Apps
that allow overlapping responses in one thread own turn scheduling and
must prevent a superseded responder from opening its first tool call after
a newer turn starts.

### Upgrade from older Turnspan versions

1. Update the browser handler to read `params.input` and return the matching
   `item_id` with `output`. Deploy it with the matching Turnspan server.
   If you wrote a custom transport, update its tool-call serialization to
   match the [protocol helpers](respond-to-user-message.md#how-request-context-flows-into-chatkit).
2. Update your store to track the new lifecycle. Let in-flight callbacks
   finish on the old release before switching traffic, or have users reload
   and restart affected interactions. Preserve completed transcript history
   and let the new server issue any new tool call with a fresh ID.
3. Verify a fresh reply and a browser callback completing its own tool
   call after reopening a durable store. Confirm that a repeated callback
   is rejected without rerunning the responder. Verify that a callback
   issued before a new message or retry cannot resume an older turn.

For a rollback, switch the browser handler, Python artifact, and store
implementation together. Keep the previous compatible storage schema or a
rollback plan for its migrations. Finish or restart in-flight tool calls on
the matching release, and reload clients still carrying the previous browser
bundle.



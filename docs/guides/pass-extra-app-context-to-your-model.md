# Pass extra app context to your model

Sometimes the model needs information that is not part of the thread: the current route, user plan, selected document, feature flags, or host-app state. This guide shows several patterns for passing that extra context into your inference pipeline.

At a high level:

- Send app/user context from the client to your ChatKit server on every request.
- Fetch context on demand with tools (including client tools).
- Inject extra context as an additional model input item when you build the request.

## Send app context with each request

### Attach headers from `useChatKit`

Use a custom `fetch` (or equivalent) when configuring `useChatKit` to attach app/user context via headers to every request:

```tsx
const chatkit = useChatKit({
  api: {
    // ...
    fetch: (input, init) => {
      const request = new Request(input, init);
      request.headers.set("X-Org-Id", currentOrgId);
      return fetch(request);
    },
  },
});
```

Read the requested organization before calling `process_request`. Resolve
membership and plan from the authenticated application session, then pass those
verified values in your request context.

```python
from dataclasses import dataclass


@dataclass
class RequestContext:
    user_id: str
    org_id: str
    plan: str
```

Pass the authenticated `RequestContext` to the server. Inside `respond`,
use `AgentContext(thread=turn.thread, store=self.store, request_context=turn.context, resume_from=None)`.
A tool can then read `ctx.context.request_context.org_id`. For data that belongs
to one agent run, declare a typed field on an `AgentContext` subclass.
The constructor reports unknown fields by name.
[Example of a typed run context](respond-to-user-message.md#save-structured-tool-results-after-a-turn)

## Fetch context on demand with tools

Sometimes you only need extra context for certain requests—fetch it on demand with tools instead of sending it for every turn.

### Server tools that fetch app context

Define a server tool that looks up app state (for example, the user’s current workspace or preferences) and returns a JSON payload to the model:

```python
from agents import RunContextWrapper, function_tool

from chatkit.agents import AgentContext


@function_tool(description_override="Fetch the user's workspace context.")
async def get_workspace_context(ctx: RunContextWrapper[AgentContext]):
    workspace = await load_workspace(ctx.context.request_context.org_id)
    return {
        "workspace_id": workspace.id,
        "features": workspace.feature_flags,
    }
```

Include this tool in your agent so the model can call it when it needs that information.

### Client tools for browser/app-only state

When the context only exists on the client (selection, viewport, local app state), use a client tool:

```python
from agents import function_tool
from agents.tool_context import ToolContext
from chatkit.agents import AgentContext


@function_tool(description_override="Read the user's current canvas selection.")
async def get_canvas_selection(ctx: ToolContext[AgentContext]) -> None:
    ctx.context.request_client_tool(
        ctx,
        name="get_canvas_selection",
        arguments={},
        continuation="local",
    )
```

On the client, implement the callback:

```ts
const chatkit = useChatKit({
  // ...
  onClientTool: async ({name, params}) => {
    const {item_id} = params;
    if (name !== "get_canvas_selection" || typeof item_id !== "string" || !item_id) {
      throw new Error("Unexpected client tool request");
    }
    const selection = myCanvas.getSelection();
    return {
      item_id,
      output: {
        nodes: selection.map((node) => ({
          name: node.name,
          description: node.description,
        })),
      },
    };
  },
});
```

Configure the agent to [stop at the client tool](update-client-during-response.md#define-a-client-tool-in-your-agent).
ChatKit sends the echoed item ID and result back to the server. After the
server claims that exact call, use `client_tool_continuation_input` to
provide the selected nodes in the next agent run.
[Build the next model request](update-client-during-response.md#build-the-next-model-request)

## Inject extra context as model input item

You can also inject context directly as an extra model input item when you build the request.

### Add a dedicated context item

Before running your agent, prepend a short, structured context item describing app/user state:

```python
from openai.types.responses import EasyInputMessageParam


extra_context = EasyInputMessageParam(
    type="message",
    role="user",
    content=(
        "<APP_CONTEXT>\n"
        f"user_id: {context.user_id}\n"
        f"org_id: {context.org_id}\n"
        f"plan: {context.plan}\n"
        "</APP_CONTEXT>"
    ),
)

input_items = [extra_context, *input_items]
```

Pair this with a short system prompt telling the model how to interpret the `<APP_CONTEXT>` block.

### Combine ids + tools

A useful pattern is to combine a lightweight context item with a follow-up tool call:

1. Add an input item that contains a stable id or handle:

   ```python
   from openai.types.responses import EasyInputMessageParam

   extra_context = EasyInputMessageParam(
       type="message",
       role="user",
       content=f"<WORKSPACE_REF id={workspace.id} />",
   )
   input_items = [extra_context, *input_items]
   ```

2. Provide a tool (server or client) that can fetch the full details when needed:

   ```python
   @function_tool(description_override="Fetch full workspace details.")
   async def fetch_workspace(ctx: RunContextWrapper[AgentContext], id: str):
       workspace = await load_workspace(id, ctx.context.request_context)
       return {
           "id": workspace.id,
           "features": workspace.feature_flags,
           "limits": workspace.limits,
       }
   ```

3. In your prompt, tell the model:

   - The `<WORKSPACE_REF>` tag carries the id it should use.
   - It should call `fetch_workspace` when it needs more detail instead of guessing.

This keeps your model inputs compact while still giving the model a reliable way to pull detailed context on demand.



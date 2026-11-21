# Tools

Tools let the assistant call into your application logic during a turn—for example to search data, run a workflow, or fetch the user’s current context—then feed the results back into the model.

At a high level:

- **Server tools** run on your backend. The assistant calls them through your inference pipeline, and you stream their results back into the conversation.
- **Client tools** run in the browser or host app. ChatKit surfaces a tool call as a streamed thread item, lets the client handle it, then resumes the conversation with the tool’s output.

## Server tools

Server tools are Python functions registered with an agent or inference pipeline. Your inference integration runs the function and returns its result to the model. With the Agents SDK, ChatKit can convert run events into updates for the conversation.

Use server tools to:

- Look up data in your own APIs or databases.
- Kick off long-running jobs while streaming progress updates.
- Update your own domain state (tickets, orders, files, etc.) in response to a turn.

From the model’s perspective, tools are structured, named capabilities it can invoke instead of guessing from free text.

## Client tools

Some operations can only run on the client—for example:

- Reading the current selection in a canvas or document.
- Inspecting local application state that never leaves the browser.
- Calling into the host app (for example, a design tool or IDE) via its own APIs.

Client tools let the model request that kind of data mid-turn:

- On the server, you instruct your inference pipeline to stop when a specific tool is called (for example, by using `StopAtTools` around that tool).
- Your responder streams a `ClientToolCallItem` and the server records its pending status.
- The client calls `onClientTool` with the tool name and `{item_id, input}`. The callback echoes that ID in `{item_id, output}`.
- The server claims the exact pending call in the authorized thread, saves the output, and calls `respond` with a [`ClientToolTurn`](../api/chatkit/turns.md#chatkit.turns.ClientToolTurn). The responder can use the stored call and output as input to the next model run.

A call can complete once. When a customer sends a new message or retries the
turn, the store cancels the pending invocation. Any late callback for that
invocation is rejected.

Use client tools when the model needs fresh, local context it cannot safely obtain from server-side state alone.

[Client tool example](../guides/update-client-during-response.md#call-client-tools-mid-inference)

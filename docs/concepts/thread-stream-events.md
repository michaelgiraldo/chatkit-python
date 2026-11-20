# Thread stream events

[`ThreadStreamEvent`](../api/chatkit/types.md#chatkit.types.ThreadStreamEvent)s are the Server-Sent Event (SSE) payloads streamed by ChatKitServer while responding to a user message or action. They keep the client UI in sync with server-side processing and drive persistence in your store.

## Thread metadata updates

ChatKitServer emits these after it creates a thread or notices metadata changes (title, status, etc.) so the UI stays in sync.

- [`ThreadCreatedEvent`](../api/chatkit/types.md#chatkit.types.ThreadCreatedEvent): introduce a new thread
- [`ThreadUpdatedEvent`](../api/chatkit/types.md#chatkit.types.ThreadUpdatedEvent): update the current thread metadata such as title or status

## Thread item events

Thread item events drive the conversation state. ChatKitServer processes these events to persist conversation state before streaming them back to the client.

- [`ThreadItemAddedEvent`](../api/chatkit/types.md#chatkit.types.ThreadItemAddedEvent): introduce a new item (message, tool call, workflow, widget, etc).
- [`ThreadItemUpdatedEvent`](../api/chatkit/types.md#chatkit.types.ThreadItemUpdatedEvent): mutate a pending item (e.g., stream text deltas, workflow task updates).
- [`ThreadItemDoneEvent`](../api/chatkit/types.md#chatkit.types.ThreadItemDoneEvent): mark an item complete and persist it.
- [`ThreadItemRemovedEvent`](../api/chatkit/types.md#chatkit.types.ThreadItemRemovedEvent): delete an item by id.
- [`ThreadItemReplacedEvent`](../api/chatkit/types.md#chatkit.types.ThreadItemReplacedEvent): swap an item in place.

Note: `ThreadItemAddedEvent` does not persist the item. `ChatKitServer` saves on `ThreadItemDoneEvent`/`ThreadItemReplacedEvent`, tracks pending items in between, and handles store writes for all `ThreadItem*Event`s.

## Errors

Stream [`ErrorEvent`](../api/chatkit/types.md#chatkit.types.ErrorEvent)s for user-facing errors in the chat UI. You can configure a custom message and whether a retry button is shown to the user.

## Progress updates

Stream [`ProgressUpdateEvent`](../api/chatkit/types.md#chatkit.types.ProgressUpdateEvent)s to show the user transient status while work is in flight.

See [Show progress while tools run](../guides/update-client-during-response.md#show-progress-while-tools-run) for more info.

## Client effects

Use [`ClientEffectEvent`](../api/chatkit/types.md#chatkit.types.ClientEffectEvent) to trigger fire-and-forget behavior on the client such as opening a dialog or pushing updates.

See [Trigger client-side effects without blocking](../guides/update-client-during-response.md#trigger-client-side-effects-without-blocking) for more info.

## Stream options

[`StreamOptionsEvent`](../api/chatkit/types.md#chatkit.types.StreamOptionsEvent) configures runtime stream behavior (for example, allowing user cancellation). `ChatKitServer` emits one at the start of every stream using `get_stream_options`; override that method to change defaults such as `allow_cancel`.

## Closing a stream

Close the ChatKit iterator when the HTTP response exits. A client can disconnect
while the transport is sending an event, between reads from the ChatKit stream.
`aiter(result)` and `result.json_events` expose the same iterator on a
`StreamingResult`. Closing an active stream releases the responder and runs the
server's cancellation hook, which saves partial assistant replies by default.

For FastAPI or Starlette, use `chatkit.fastapi.process_request` to close the
stream when the HTTP response ends.
[FastAPI adapter reference](../api/chatkit/fastapi.md)
It keeps the iterator open for the lifetime of the HTTP response and closes it
when the framework reports a disconnect.

When your own `respond` or `action` generator forwards another async generator,
wrap the inner generator in `contextlib.aclosing`. This carries early closure
through the forwarding generator, including when it is suspended at a `yield`.

## Related guides
- [Respond to a user message](../guides/respond-to-user-message.md)

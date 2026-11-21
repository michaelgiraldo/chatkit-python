# Thread stream events

[`ThreadStreamEvent`](../api/chatkit/events.md#chatkit.events.ThreadStreamEvent)s are the Server-Sent Event (SSE) payloads streamed by ChatKitServer while responding to a user message or action. They keep the client UI in sync with server-side processing and drive persistence in your store.

## Thread metadata updates

ChatKitServer emits a creation event when it saves a new thread. To save metadata changes such as a title or status and show them in the UI, apply a patch with `ChatKitServer.update_thread` and yield the returned event.

- [`ThreadCreatedEvent`](../api/chatkit/events.md#chatkit.events.ThreadCreatedEvent): introduce a new thread
- [`ThreadUpdatedEvent`](../api/chatkit/events.md#chatkit.events.ThreadUpdatedEvent): update the current thread metadata such as title or status

## Thread item events

Thread item events drive the conversation state. ChatKitServer processes these events to persist conversation state before streaming them back to the client.

- [`ThreadItemAddedEvent`](../api/chatkit/events.md#chatkit.events.ThreadItemAddedEvent): introduce a new item (message, tool call, workflow, widget, etc).
- [`ThreadItemUpdatedEvent`](../api/chatkit/events.md#chatkit.events.ThreadItemUpdatedEvent): mutate a pending item (e.g., stream text deltas, workflow task updates).
- [`ThreadItemDoneEvent`](../api/chatkit/events.md#chatkit.events.ThreadItemDoneEvent): mark an item complete and persist it.
- [`ThreadItemRemovedEvent`](../api/chatkit/events.md#chatkit.events.ThreadItemRemovedEvent): delete an item by id.
- [`ThreadItemReplacedEvent`](../api/chatkit/events.md#chatkit.events.ThreadItemReplacedEvent): swap an item in place.

The server tracks additions and deltas in memory while the item streams. A done event inserts the completed item, a replacement event saves the new value, and a removal event deletes it. On cancellation, the default handler saves pending assistant messages with content.

## Errors

Stream [`ErrorEvent`](../api/chatkit/events.md#chatkit.events.ErrorEvent)s for user-facing errors in the chat UI. You can configure a custom message and whether a retry button is shown to the user.

## Progress updates

Stream [`ProgressUpdateEvent`](../api/chatkit/events.md#chatkit.events.ProgressUpdateEvent)s to show the user transient status while work is in flight.

See [Show progress while tools run](../guides/update-client-during-response.md#show-progress-while-tools-run) for more info.

## Client effects

Use [`ClientEffectEvent`](../api/chatkit/events.md#chatkit.events.ClientEffectEvent) to trigger fire-and-forget behavior on the client such as opening a dialog or pushing updates.

See [Trigger client-side effects without blocking](../guides/update-client-during-response.md#trigger-client-side-effects-without-blocking) for more info.

## Stream options

[`StreamOptionsEvent`](../api/chatkit/events.md#chatkit.events.StreamOptionsEvent) configures stream behavior before response events begin. `ChatKitServer` gets these options from `get_stream_options`. Override that method to change defaults such as `allow_cancel`.

## Closing a stream

Close the ChatKit iterator when the HTTP response exits. A client can disconnect
while the transport is sending an event, between reads from the ChatKit stream.
`StreamingResult` is an async iterator of typed events. For a custom transport,
own it with `async with aclosing(result)` and encode each event with
`chatkit.protocol.encode_sse`. Closing an active stream releases the responder
and saves partial assistant replies by default. If the stream closes before
it finishes a client tool handoff, ChatKit cancels the exact pending call
it opened. A completed handoff leaves the call pending for the browser.
[Client tool lifecycle](../guides/update-client-during-response.md#know-when-a-callback-can-complete)

For FastAPI or Starlette, use `chatkit.fastapi.process_request`. Its default
streaming response listens for `http.disconnect` while it waits for the next
event, so a visitor can leave even when the producer is silent. On servers
advertising ASGI HTTP spec 2.4 or later, a failed write can also signal that
the connection closed. [ASGI HTTP specification](https://asgi.readthedocs.io/en/stable/specs/www.html#disconnected-client-send-exception)
· [FastAPI adapter reference](../api/chatkit/fastapi.md)

When your own `respond` or `action` generator forwards another async generator,
wrap the inner generator in `contextlib.aclosing`. This carries early closure
through the forwarding generator, including when it is suspended at a `yield`.

## Related guides
- [Respond to a user message](../guides/respond-to-user-message.md)

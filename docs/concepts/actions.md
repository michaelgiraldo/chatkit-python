# Actions

ChatKit actions are interaction events triggered by widgets or client code that let the client and server run logic or start a model response independently of user messages.

Every action needs a non-empty `type`. The browser can omit `payload` for an action
with no arguments. When constructing a typed `Action` in Python, pass the declared
payload explicitly, including `None` if the action accepts it.

## Widget actions

Widget actions are specified in the widget definition itself (for example, `Button.onClickAction`), so every interaction carries a typed action payload plus the widget item that fired it. By default actions are routed to the server, but you can set `handler: "client"` when you want to intercept the action in the browser first.

### Server-handled actions

If you leave the handler unset, the action is delivered to `ChatKitServer.action(thread, action, sender, context)`, where `sender` is the widget item that triggered it when that item is available. Server handling is the right choice when you need to mutate thread state, stream widget or message updates, or start an agent response without a new user message. Record important interactions as hidden context so the model can react on the next turn (for example, “user clicked confirm”), and treat `action.payload` as untrusted input that must be validated and authorized before you persist anything.

### Client-handled actions

When you set `handler: "client"`, the action flows into the client SDK’s `widgets.onAction` callback so you can do immediate UI work such as opening dialogs, navigating, or running local validation. Client handlers can still forward a follow-up action to the server with `chatkit.sendCustomAction()` after local logic finishes. The server thread stays unchanged unless you explicitly send that follow-up action or a message.

### Sync server-handled actions

To run a widget action while a response is streaming, set `"streaming": false` in JSON or `streaming=False` in Python. The client sends a non-streaming request to `ChatKitServer.sync_action(thread, action, sender, context)`.

Return a [`SyncCustomActionResponse`](../api/chatkit/events.md#chatkit.events.SyncCustomActionResponse) with `updated_item` to replace the item in the browser. Call `await self.store.save_item(thread.id, updated_item, context)` if the replacement should survive a reload.

Return `SyncCustomActionResponse()` when the action has no visible update. If the handler returns a `HiddenContextItem` or `SDKHiddenContextItem`, the server sends the same empty JSON reply (`{}`).

## Client-sent actions using the chatkit.sendCustomAction() command

Your client integration can also initiate actions directly with `chatkit.sendCustomAction(action, itemId?)`, optionally namespaced to a specific widget item. The server receives these in `ChatKitServer.action` just like a widget-triggered action and can stream widgets, messages, or client effects in response. This pattern is useful when a flow starts outside a widget—or after a client-handled action—but you still want the server to persist results or involve the model.

## Related guides

- [Build interactive responses with widgets](../guides/build-interactive-responses-with-widgets.md)

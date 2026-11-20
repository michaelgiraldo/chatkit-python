# Chatkit Python API Reference

What you'll find here:

- **Core server API**: [`chatkit.server`](server.md) – `ChatKitServer` and related helpers for receiving messages, streaming responses, and wiring up tools and widgets.
- **Actions**: [`chatkit.actions`](actions.md) – `Action` and `ActionConfig` for widget callbacks.
- **FastAPI adapter**: [`chatkit.fastapi`](fastapi.md) – HTTP request handling and disconnect cleanup for the optional `fastapi` extra.
- **Protocol**: [`chatkit.protocol`](protocol.md) – JSON request parsing and encoding for transport adapters.
- **Turns**: [`chatkit.turns`](turns.md) – typed triggers, request context, and response outcomes.
- **Store integration**: [`chatkit.store`](store.md) – separate interfaces for threads and items, attachment metadata, and attachment file operations.
- **Store checks**: [`chatkit.store_checks`](store_checks.md) – contract checks for app-owned thread stores.
- **Agents**: [`chatkit.agents`](agents.md) streams provider output. [`chatkit.agent_input`](agent_input.md) converts saved history into model input.
- **Data models**: [`chatkit.models`](models.md) defines threads, items, and attachments.
- **Requests and events**: [`chatkit.requests`](requests.md) describes browser operations. [`chatkit.events`](events.md) defines streamed updates.
- **Errors**: [`chatkit.errors`](errors.md) – structured error types your ChatKit integration can raise so ChatKit can emit consistent `ErrorEvent`s to the client.
- **Widgets**: [`chatkit.widgets`](widgets.md) provides typed builders. [`chatkit.widget_models`](widget_models.md) owns the JSON tree, [`chatkit.widget_stream`](widget_stream.md) streams updates, and [`chatkit.templates`](templates.md) loads `.widget` files with the `templates` extra.

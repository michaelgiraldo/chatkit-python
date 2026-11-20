# Chatkit Python API Reference

What you'll find here:

- **Core server API**: [`chatkit.server`](server.md) – `ChatKitServer` and related helpers for receiving messages, streaming responses, and wiring up tools and widgets.
- **FastAPI adapter**: [`chatkit.fastapi`](fastapi.md) – HTTP request handling and disconnect cleanup for the optional `fastapi` extra.
- **Protocol**: [`chatkit.protocol`](protocol.md) – JSON request parsing and encoding for transport adapters.
- **Turns**: [`chatkit.turns`](turns.md) – typed triggers, request context, and response outcomes.
- **Store integration**: [`chatkit.store`](store.md) – separate interfaces for threads and items, attachment metadata, and attachment file operations.
- **Agents integration helpers**: [`chatkit.agents`](agents.md) – helpers and utilities for using ChatKit together with the Agents SDK.
- **Data models and types**: [`chatkit.types`](types.md) – Pydantic models for threads, items, events, and other shared types.
- **Errors**: [`chatkit.errors`](errors.md) – structured error types your ChatKit integration can raise so ChatKit can emit consistent `ErrorEvent`s to the client.
- **Widgets**: [`chatkit.widgets`](widgets.md) – named builders, `WidgetTemplate`, and the canonical `WidgetRoot` and `WidgetComponent` models for rich UI responses.

# Release process/changelog

The project follows a slightly modified version of [semantic versioning](https://semver.org/spec/v2.0.0.html). The SDK is still evolving and certain backwards-incompatible changes may be released as minor versions.

See [Turnspan's GitHub releases](https://github.com/michaelgiraldo/chatkit-python/releases)
and the original [OpenAI ChatKit releases](https://github.com/openai/chatkit-python/releases).

## Minor versions

We will increase minor versions for **breaking changes** to any public interfaces. For example, going from `1.0.x` to `1.1.x` might include breaking changes.

If you don't want breaking changes, we recommend pinning to `1.0.x` versions in your project.

## Patch versions

We will increment patch versions for non-breaking changes:

- Bug fixes
- New features
- Changes to private interfaces

## Breaking change changelog

### Unreleased

- Client tool callbacks use `AgentContext.request_client_tool` with the
  Agents SDK's `ToolContext` to bind the browser callback to its invocation.
- Custom HTTP transports parse the request with `chatkit.protocol.parse_request`,
  pass the typed request to `ChatKitServer.process`, and serialize the typed
  response or closeable event stream with the protocol helpers.
- Thread stores implement atomic `claim_pending_item`, `patch_thread`, and
  `truncate_after` operations. Inside a responder, yield the event from
  `ChatKitServer.update_thread` to persist a display change and show it to
  the client. Use `ThreadStore.patch_thread` for metadata-only writes.
- Typed widget builders use `.build()` to produce a canonical widget tree
  for direct item or update construction. Use `WidgetTemplate.build()` for
  every template root, including entity previews.
- Install the `agents` extra when using `chatkit.agents`. The `fastapi` extra
  supplies the HTTP adapter's dependencies and Uvicorn.

### Turnspan 2 (`1.6.5+turnspan.2`)

- Lists inside cards now parse as typed `ListView` widgets, including their rows
  and click actions. Thanks to Saloni Garg for the [upstream fix](https://github.com/openai/chatkit-python/pull/137).
- Chat Completions replies now get distinct message IDs. Each ID follows its
  reply from streaming through reload. Responses keeps the IDs supplied by
  OpenAI. Thanks to Jacob Babula for the [upstream fix](https://github.com/openai/chatkit-python/pull/128).
- Requests made with the Agents SDK now use the SDK's default User-Agent or the
  app's configured value. Turnspan no longer adds a `ChatKit/Python` suffix.
  The dependency range remains at Agents SDK 0.25.x pending tests of the next
  minor release.
- Plain `pytest` works in a clean checkout with `PYTHONPATH` unset, using the
  same configuration as `make test`. Thanks to Mark Zhang for
  [reporting the test import problem](https://github.com/openai/chatkit-python/pull/122).
- The [structured feedback example](guides/respond-to-user-message.md#save-structured-tool-results-after-a-turn)
  shows how tools can access the thread ID and how the app can save feedback
  after a successful stream. Thanks to Jao Garcia for the
  [persistence question](https://github.com/openai/chatkit-python/issues/120).

### Turnspan 1 (`1.6.5+turnspan.1`)

- The fork now requires Python 3.14 or newer. For Python 3.10 through 3.13, the upstream `v1.6.5` release remains available.
- Installing the fork requires OpenAI Python 3.56.0 through 3.x, Agents SDK
  0.25.x, Pydantic 2.14.4 through 2.x, Uvicorn 0.54.0 through 0.x, and Jinja
  3.1.6 through 3.x. Turnspan 1 uses internal per-request header overrides from
  Agents SDK 0.25.x. Update any older pins in your app before installing the fork.
  If your app passes a custom HTTP client to the OpenAI SDK, see the
  [HTTPX2 migration guide](https://github.com/openai/openai-python/blob/v3.56.0/httpx2.md).
- The `fastapi` extra installs the FastAPI version used by the quickstart.
- Interrupted streams close cleanly. The FastAPI example verifies that a
  partial reply can be reloaded and the thread can accept another turn.
- Widget updates handle added and removed properties and text without IDs.
  Repeated workflow tasks update the correct position, and empty streamed
  citations leave subsequent citations intact.

### 1.6.0

- `Store.save_attachment` may now be called with updates to existing attachments (treated as an upsert), not only inserts.

### 1.5.0

Two-phase uploads:

- `upload_url` was removed from `FileAttachment` and `ImageAttachment`; use `upload_descriptor` instead.
- `ChatKitServer` now saves the created attachment metadata in the store when handling the `attachments.create` request; remove the store-write step in `AttachmentStore.create_attachment`.

### 1.4.0

- Widget and action classes are still usable but marked as deprecated in favor of using `WidgetTemplate` to build widgets from `.widget` files.
- Added `jinja2` as a required dependency for widget template rendering.
- A stop button is now shown by default during streaming, allowing users to cancel the stream mid-response. Integrations can override `ChatKitServer.get_stream_options` to change this behavior.

### 1.3.0

- Fixed the type for the `defaultChecked` property of `Checkbox` widgets, updating it from `string` to `bool`.

### 1.2.0

- Updated `agents.stream_agent_response` to add annotation parts as they are received rather than adding all the annotations at the end after the response is completed.
- Added support for rendering `container_file_citation`.

### 1.1.0

- `CustomSummary`, `CustomTask`, and `EntitySource` types have been updated to restrict `icon` to `IconName`.
- All `ThreadItemConverter` methods have been updated to be asynchronous.

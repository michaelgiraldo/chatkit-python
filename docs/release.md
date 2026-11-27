# Releases and changelog

Turnspan keeps the OpenAI ChatKit base version and increments the
`+turnspan.N` suffix for fork builds. Get published builds from
[Turnspan's GitHub releases](https://github.com/michaelgiraldo/chatkit-python/releases).
The `openai-chatkit` package on PyPI is published by the
[original OpenAI project](https://github.com/openai/chatkit-python/releases).

## Verify and publish a release

1. Set the version in `pyproject.toml`, refresh `uv.lock`, and add a matching
   changelog section. Commit the candidate and run `make check`, `make build`,
   and `make release-check` with Python 3.14. The release check installs the
   committed `HEAD` in isolation and exercises the quickstart, FastAPI, and
   Agents SDK with synthetic inputs.
2. Merge to `main` and wait for CI to pass. Run the **Verify Turnspan release**
   workflow once with the full commit SHA. It verifies that commit's CI result
   and repeats the consumer check using the public GitHub source archive.

   ```sh
   gh workflow run turnspan-release.yml --ref main -f sha=COMMIT_SHA
   ```

3. Copy the release notes from the workflow summary. Create the GitHub release
   manually with the indicated tag and exact commit. If the tag already exists,
   verify its target. After publishing, confirm the tag resolves to the tested
   commit and update the shared install section in the README. The docs
   landing page renders that same section.

The workflow only verifies and produces notes. The notes include the commit,
install command, tested dependency versions, and links to docs at that commit.

When changing the Agents SDK range, test the supported minimum and the newest
supported release before publishing. CI tests the version in `uv.lock`; use
`make sdk-check AGENTS_VERSION=VERSION` to test a different supported version.
The source-archive check verifies what a fresh consumer install resolves.

## Turnspan 9 (`1.6.5+turnspan.9`)

This is a dependency refresh. App dependency ranges, Python 3.14 minimum,
browser protocol, and thread-store contracts are unchanged from Turnspan 8.

- Update the locked [OpenAI Python SDK](https://github.com/openai/openai-python/releases/tag/v3.60.0)
  from 3.57.0 to 3.60.0. The declared range stays `>=3.57.0,<4`.
  The Agents SDK stays at 0.26.0 (`>=0.26.0,<0.27`).
- Update [Zensical](https://github.com/zensical/zensical/releases/tag/v0.1.4)
  to 0.1.4 and refresh Certifi, charset-normalizer, Markdown, packaging,
  platformdirs, PyMdown Extensions, PyYAML, sse-starlette, watchdog, and
  Zensical UI in the lock.
- Update the pinned [setup-uv action](https://github.com/astral-sh/setup-uv/releases/tag/v10.4.0)
  to 10.4.0. Checkout, Pages actions, and the upstream-only PyPI publisher
  were already current. The fork still publishes through GitHub Releases.
- Review the Pydantic core, FastAPI/Starlette/AnyIO/Uvicorn, Jinja2, setuptools,
  and test/type/lint tool requirements. No direct update was available for
  those packages within their existing requirements.
- The model adapter still uses the Responses and Chat Completions SDK types.
  No migration is needed for this release under the
  [OpenAI deprecation schedule](https://developers.openai.com/api/docs/deprecations).
  The demo loads the OpenAI-managed browser component instead of an npm
  dependency and uses Turnspan's own endpoint, not hosted Agent Builder.
- The OSV audit of the 73 locked dependencies found no known advisories on
  Linux, macOS, or Windows for Python 3.14. The SDK suite passed on Python
  3.14 and 3.15, and on the OpenAI Python 3.57.0 floor. The local browser
  demo also passed message, widget action, and history reload checks.

## Turnspan 8 (`1.6.5+turnspan.8`)

- **New Agents SDK baseline.** The `agents` extra requires
  `openai-agents>=0.26.0,<0.27`. If your app pins 0.25, update that pin and
  refresh your lock before adopting Turnspan 8. Run your streaming, history,
  and tool-approval checks against the new lock. The core install without the
  `agents` extra is unaffected. No browser or storage migration is required
  from Turnspan 7.
- `check_thread_store` and `check_client_tool_store` accept a `context_factory`
  for stores with mutable request context. Each simulated request can get fresh
  state for the same authenticated visitor. Existing callers can omit it.
- The [production guide](guides/prepare-your-app-for-production.md) covers
  signed thread-list smoke tests and delayed and repeated retention events.

## Turnspan 7 (`1.6.5+turnspan.7`)

### Upgrading an app from Turnspan 6

Update the backend source pin and refresh your app's dependency lock. Python
3.14 remains the minimum. You can keep your existing ChatKit browser component.
Check the rows that apply to your backend.

| If your app uses… | What to change or verify |
| --- | --- |
| A custom thread store | For browser reads (`include_hidden=False`), filter server-only items and ineligible pending browser tools **before** applying the cursor and limit. Compute `after` and `has_more` from visible items. Turnspan now rejects a page containing either hidden context type. Apply the same rule if your responder supplies paginated thread snapshots. [Store guide](guides/respond-to-user-message.md#implement-your-chatkit-data-store) |
| Forms | Record which turn issued the form. In the same transaction that claims it, raise `ConflictError` and leave it pending if a different turn is `ACCEPTED` or `RUNNING`. This prevents an old form submission from interrupting a newer reply. [SQLite example](examples/sqlite_thread_store.py) |
| Extra fields on `AgentContext` | Unknown constructor keywords now raise a validation error. Move request-scoped values into `request_context` or declare typed fields on a subclass. [App context guide](guides/pass-extra-app-context-to-your-model.md#send-app-context-with-each-request) |
| Agents SDK tools that pause for human approval | Implement `ChatKitServer.save_approval_checkpoint(thread, turn_id, checkpoint, context)`. Save the serialized run privately under the authenticated owner, thread, and turn. Forward `ApprovalRequired` if your responder filters events. A successful save puts the turn in `WAITING`. A missing hook or failed save produces `FAILED`. [Approval example](guides/respond-to-user-message.md#pause-for-tool-approval) |
| Store tests for browser tools | Run both `check_thread_store` and `check_client_tool_store`. The first now covers core storage and browser history. The second exercises browser tool callbacks. [Store test example](guides/respond-to-user-message.md#check-your-apps-store) |
| An import of `chatkit.widgets.MinMax` | Remove the import. If your app used this removed annotation for its own data, define the bounds type in your app. Chart JSON schemas are unchanged. |
| The `agents` extra | Allow `openai>=3.57.0,<4` in your app lock. The Agents SDK requirement remains `openai-agents>=0.25.0,<0.26`. Zensical 0.1.3 is a documentation build dependency. |

### Streaming and thread history

- When a visitor leaves during a silent FastAPI stream, the server stops the
  producer. On ordinary streaming turns the partial reply survives reload, and
  the visitor can send another message.
- If a responder streams `Hi` and then finishes building that same message,
  the saved reply reads `Hi` once. Turnspan also isolates streamed citations
  and workflow tasks from subsequent edits the responder hasn't sent.
- Widget actions keep app and SDK hidden context out of browser replies.
  A private context update leaves an active progress display open while the
  agent continues working.
- When the app replays thread history, Turnspan sends assistant text as model
  input and keeps the recorded Responses phase, such as `commentary` or
  `final_answer`. Stores that serialize individual fields should round-trip
  the optional assistant `phase` to preserve this behavior.

### Optional reviewed replies

For an app that checks an answer before a visitor sees it, configure an Agents
SDK output guardrail and pass `delivery="reviewed_text"` to
`stream_agent_response`. The browser receives the approved final text in one
message. A rejection or disconnect during review keeps the candidate out of
the ChatKit transcript. Ordinary streaming remains the default for apps that
show intermediate events or use interactive browser tools.
[Reviewed delivery guide](guides/respond-to-user-message.md#deliver-reviewed-text)

## Turnspan 6 (`1.6.5+turnspan.6`)

- **Breaking store contract.** Every streaming response has a durable turn
  ID. Custom stores implement atomic turn ownership, fenced responder
  writes, and persisted milestones. Use a fresh schema for Turnspan 6.
  Streaming actions receive `turn_id`. Supply the current ID to
  `update_thread` and other direct responder writes.
- The server distinguishes a saved callback from responder start and
  completion. Closing a response before processing starts records
  `ABANDONED`. Started responders can report `SUPERSEDED` when a new
  request takes ownership.
- **Breaking Python and storage contract.** Browser tools publish a
  `ClientToolCallItem` containing display data. The server registers a
  separate `ClientToolInvocation` with model data and the turn linkage.
  Custom responders yield `ClientToolRequest` to register the pair.
  Pass the complete `ClientToolTurn` to
  `client_tool_continuation_input`. The ChatKit browser still sends the
  same callback envelope.
- Browser tools return the ID of the exact saved invocation. Each callback can
  complete once, and a newer turn can cancel a pending invocation.
- The Agents SDK adapter preserves the original model call and any sibling tool
  results. Choose local replay or OpenAI-managed Responses context when
  registering the tool.
- The [browser tool guide](guides/update-client-during-response.md#call-client-tools-mid-inference)
  documents the current callback, continuation, and storage contracts.

## Turnspan 5 (`1.6.5+turnspan.5`)

- Development milestone. `AgentContext` takes an explicit `resume_from` option
  for a saved display workflow. Use `None` for a fresh run.

## Turnspan 4 (`1.6.5+turnspan.4`)

- Development milestone. Added control over workflow history reads.

## Turnspan 3 (`1.6.5+turnspan.3`)

- Early published build. Added typed turns, explicit store operations, optional
  integrations, a FastAPI adapter, and a runnable local demo.
- Improved streamed and reloaded widgets. Image previews and web-search progress
  use provider item IDs. The web-search work builds on the
  [upstream report and proposed fix](https://github.com/openai/chatkit-python/pull/29).
- Added the durable SQLite example, custom-store checks, and generated docs.
  Thanks to Mark Zhang for the [test import report](https://github.com/openai/chatkit-python/pull/122).

## Turnspan 2 (`1.6.5+turnspan.2`)

- Fixed nested [ListView widgets](https://github.com/openai/chatkit-python/pull/137)
  and preserved distinct [Chat Completions message IDs](https://github.com/openai/chatkit-python/pull/128)
  across streaming and reload.
- Added a [structured feedback example](guides/respond-to-user-message.md#save-structured-tool-results-after-a-turn)
  following an [upstream persistence question](https://github.com/openai/chatkit-python/issues/120).

## Turnspan 1 (`1.6.5+turnspan.1`)

- Established the Python 3.14 baseline and improved stream cleanup, widget
  updates, workflow task indexes, and streamed citation handling.

[OpenAI ChatKit release history](https://github.com/openai/chatkit-python/releases)

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

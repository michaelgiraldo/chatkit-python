# Turnspan for ChatKit

<!-- --8<-- [start:start-here] -->
Turnspan is a maintained fork of [OpenAI's ChatKit Python SDK](https://github.com/openai/chatkit-python)
for self-hosted ChatKit applications. It connects the official ChatKit browser
component to a Python backend you control.

Coming from OpenAI's repository? Install `openai-chatkit` from this fork
and import its modules through `chatkit`. Turnspan requires Python 3.14 or
newer. Application responders implement `respond(self, turn: Turn)`.
The Agents SDK and FastAPI adapters are optional extras.
Use a fresh thread store for your first setup. The
[server guide](https://michaelgiraldo.github.io/chatkit-python/guides/respond-to-user-message/)
shows how to wire the responder and storage.

## Try the demo

With Git, `uv`, and `make` installed, clone the fork and run the demo.
If you already have the checkout you want to try, run `make demo` there.

```sh
git clone https://github.com/michaelgiraldo/chatkit-python.git turnspan-demo
cd turnspan-demo
make demo
```

Open [the demo on port 8000](http://127.0.0.1:8000/) and send `hello`.
The demo uses sample replies and runs without a model API key.

## Install Turnspan in an app

[Turnspan 9](https://github.com/michaelgiraldo/chatkit-python/releases/tag/v1.6.5%2Bturnspan.9)
is the latest stable release (`openai-chatkit 1.6.5+turnspan.9`). Install
the verified source commit in your app's Python environment.

```sh
python -m pip install 'openai-chatkit[agents,fastapi] @ https://github.com/michaelgiraldo/chatkit-python/archive/ccddcf0706e2beaaeb8c74a72ebeeb72775157f4.tar.gz'
```

Omit `[agents,fastapi]` for the core server alone. Add `templates` to the
extras if your app loads `.widget` files. See the
[docs at the release commit](https://github.com/michaelgiraldo/chatkit-python/tree/ccddcf0706e2beaaeb8c74a72ebeeb72775157f4/docs)
for the matching Python and storage contracts.

**Upgrading from Turnspan 7?** Apps using the `agents` extra need Agents SDK
0.26. Update any 0.25 pin and refresh your app's lockfile. The browser component
and the storage contract are unchanged for this upgrade.

**Coming from Turnspan 6?** Also check your thread store's history queries
and form handling. Apps using Agents SDK tool approvals also need to save
paused runs. The [app migration checklist](https://michaelgiraldo.github.io/chatkit-python/release/#upgrading-an-app-from-turnspan-6)
covers the affected integrations. The ChatKit browser component can stay on
its current version for this upgrade.

To install a local development checkout, run this from its repository root.

```sh
python -m pip install '.[agents,fastapi]'
```
<!-- --8<-- [end:start-here] -->

## Documentation

Use the [rendered documentation from main](https://michaelgiraldo.github.io/chatkit-python/)
for guides and the [generated API reference](https://michaelgiraldo.github.io/chatkit-python/api/chatkit/).
Use the [release notes](https://github.com/michaelgiraldo/chatkit-python/releases)
and their pinned source links for version-specific instructions.

## Develop Turnspan

The [changelog](docs/release.md) records the current API changes. The
[project scope](docs/index.md#project-scope) explains what the SDK and each
application own.

The fork's `main` branch is the maintained Turnspan branch. Its CI workflow
publishes the rendered documentation after checks pass.

Run `make sync` to install the locked development environment and `make check`
to run formatting, lint, type checks, tests, and the strict docs build. Use
`make test TESTS='tests/test_fastapi.py -k missing_thread'` for a focused run.

To test OpenAI Python 3.57.0 without changing the lockfile, run
`make sdk-check OPENAI_VERSION=3.57.0`. Set `AGENTS_VERSION` to test the
Agents SDK, or supply both versions to try a pair. An omitted SDK uses its
locked version. The command reports the installed versions and runs the full
test suite in an isolated environment with synthetic provider traffic.

When changing modules, imports, or optional dependencies, run `make wheel-check`.
It builds a fresh package and checks the installed core, FastAPI, Agents, and
template imports in isolated environments. CI runs the same command.

Run `make serve-docs` to preview the Zensical site and `make docs-check` to
build it into `site/`. Zensical generates the API reference from `chatkit/`.
Add new public modules to the API navigation in [zensical.toml](zensical.toml).

### Where to start a fix

| Symptom | Code to read first | Regression to run |
| --- | --- | --- |
| A reply is doubled or missing after a disconnect | [Partial-item assembly](chatkit/_pending_items.py) builds the text. [Server cancellation](chatkit/server.py) decides what to save. | [Partial replies](tests/test_stream_lifecycle.py) and [disconnect, reload, and next turn](tests/test_http_disconnect.py) |
| A private item appears in chat or a thread reopens with missing messages | [Browser output](chatkit/_browser_output.py) checks visibility. [Request handling](chatkit/server.py) reads the first page and handles continuation requests. | [Browser history requests](tests/test_fastapi.py), especially `test_browser_reopens_thread_and_continues_from_first_page_cursor` |
| A provider citation, image preview, or reply ID is wrong | [Agent output conversion](chatkit/_agent_output.py) turns provider events into ChatKit items. | [Output conversion](tests/test_agent_output.py) and [both SDK model paths](tests/test_openai_sdk_stream.py) |
| A custom store skips or repeats items across pages | [ThreadStore](chatkit/store.py) defines cursor behavior. [Store checks](chatkit/store_checks.py) exercise the app’s implementation. | [Store regressions](tests/test_store.py), especially `test_core_store_check_detects_repeated_visible_page` |

## License
This project is licensed under the [Apache License 2.0](LICENSE).

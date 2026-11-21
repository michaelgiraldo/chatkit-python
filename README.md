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

[Turnspan 6](https://github.com/michaelgiraldo/chatkit-python/releases/tag/v1.6.5%2Bturnspan.6)
is the latest stable release (`openai-chatkit 1.6.5+turnspan.6`). Install it
from the verified source commit in your app's Python environment.

```sh
python -m pip install 'openai-chatkit[agents,fastapi] @ https://github.com/michaelgiraldo/chatkit-python/archive/0d916c03ef8e48d4d77ddcc461431e1d6f7b8718.tar.gz'
```

Omit `[agents,fastapi]` for the core server alone. Add `templates` to the
extras if your app loads `.widget` files. See the
[docs at the release commit](https://github.com/michaelgiraldo/chatkit-python/tree/0d916c03ef8e48d4d77ddcc461431e1d6f7b8718/docs)
for the matching Python and storage contracts.

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

When changing modules, imports, or optional dependencies, run `make wheel-check`.
It builds a fresh package and checks the installed core, FastAPI, Agents, and
template imports in isolated environments. CI runs the same command.

Run `make serve-docs` to preview the Zensical site and `make docs-check` to
build it into `site/`. Zensical generates the API reference from `chatkit/`.
Add new public modules to the API navigation in [zensical.toml](zensical.toml).

## License
This project is licensed under the [Apache License 2.0](LICENSE).

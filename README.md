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

These docs describe the **Turnspan 6 release candidate**
(`openai-chatkit 1.6.5+turnspan.6`). To try a reviewed development checkout,
run the following from its repository root in your app's Python environment.

```sh
python -m pip install '.[agents,fastapi]'
```

Use `.` for the core server alone. Add `templates` to the extras if your
app loads `.widget` files.

[Turnspan 3](https://github.com/michaelgiraldo/chatkit-python/releases/tag/v1.6.5%2Bturnspan.3)
is the latest published development build. Install it from its tested
source commit and use [the docs for that commit](https://github.com/michaelgiraldo/chatkit-python/tree/e0decabf3101e4326af054ebfe2c766112e30e66/docs).

```sh
python -m pip install 'openai-chatkit[agents,fastapi] @ https://github.com/michaelgiraldo/chatkit-python/archive/e0decabf3101e4326af054ebfe2c766112e30e66.tar.gz'
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

Run `make serve-docs` to preview the Zensical site and `make docs-check` to
build it into `site/`. Zensical generates the API reference from `chatkit/`.
Add new public modules to the API navigation in [zensical.toml](zensical.toml).

## License
This project is licensed under the [Apache License 2.0](LICENSE).

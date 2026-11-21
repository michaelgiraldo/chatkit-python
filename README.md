# Turnspan for ChatKit

Turnspan is a maintained Python 3.14 fork of OpenAI ChatKit.

<!-- --8<-- [start:stable-start] -->
## Install Turnspan in an app

[Turnspan 3](https://github.com/michaelgiraldo/chatkit-python/releases/tag/v1.6.5%2Bturnspan.3)
is the latest published release. Run this command in your app's Python 3.14
environment to install the package with the Agents SDK and FastAPI adapters.

```sh
python -m pip install 'openai-chatkit[agents,fastapi] @ https://github.com/michaelgiraldo/chatkit-python/archive/e0decabf3101e4326af054ebfe2c766112e30e66.tar.gz'
```

Use the same URL with `openai-chatkit` for the core server alone. Add
`templates` inside the brackets if your app loads `.widget` files.
For a locked app dependency and deployment verification, see the
[Turnspan 3 pinning guide](https://github.com/michaelgiraldo/chatkit-python/blob/e0decabf3101e4326af054ebfe2c766112e30e66/docs/guides/pin-the-fork.md).

## Try the stable demo

With Git, `uv`, and `make` installed, check out the published tag to get the
demo source. The demo uses sample replies and runs without a model API key.

```sh
git clone https://github.com/michaelgiraldo/chatkit-python.git turnspan-demo
cd turnspan-demo
git switch --detach 'v1.6.5+turnspan.3'
make demo
```

Open [the demo on port 8000](http://127.0.0.1:8000/) and send `hello`.
<!-- --8<-- [end:stable-start] -->

## Documentation

Use the [rendered documentation from main](https://michaelgiraldo.github.io/chatkit-python/)
for guides and the [generated API reference](https://michaelgiraldo.github.io/chatkit-python/api/chatkit/).
Use the [release notes](https://github.com/michaelgiraldo/chatkit-python/releases)
and their pinned source links for version-specific instructions.

## Develop Turnspan

<!-- --8<-- [start:development-status] -->
The code documented here is the **Turnspan 6 release candidate**
(`openai-chatkit 1.6.5+turnspan.6`).
<!-- --8<-- [end:development-status] -->

The [changelog](docs/release.md#turnspan-6-165turnspan6)
and [Turnspan 3 to 6 migration guide](docs/guides/upgrade-turnspan-3-to-6.md)
describe the candidate's breaking changes. The [project scope](docs/index.md#project-scope)
explains what the SDK and each application own.

Install this checkout with `python -m pip install .` to try its code. The
fork's `main` branch is the maintained Turnspan branch. Its CI workflow
publishes the rendered documentation after checks pass.

Run `make sync` to install the locked development environment and `make check`
to run formatting, lint, type checks, tests, and the strict docs build. Use
`make test TESTS='tests/test_fastapi.py -k missing_thread'` for a focused run.

Run `make serve-docs` to preview the Zensical site and `make docs-check` to
build it into `site/`. Zensical generates the API reference from `chatkit/`.
Add new public modules to the API navigation in [zensical.toml](zensical.toml).

## License
This project is licensed under the [Apache License 2.0](LICENSE).

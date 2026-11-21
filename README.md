# Turnspan for ChatKit

Turnspan is a maintained Python 3.14 fork of OpenAI ChatKit.

**Turnspan 6** installs as `openai-chatkit 1.6.5+turnspan.6` and requires
Python 3.14 or newer. See the [breaking changes](docs/release.md#turnspan-6-165turnspan6)
and [upgrade guide](docs/guides/update-client-during-response.md#upgrade-from-older-turnspan-versions)
for matching browser callbacks to exact saved tool calls.

Install from the root of the checkout with `python -m pip install .`. Select
the `agents` extra for `chatkit.agents`, `fastapi` for the HTTP adapter and
Uvicorn, and `templates` to load `.widget` files.

Use the [quickstart](docs/quickstart.md) to run a server or the
[pinning guide](docs/guides/pin-the-fork.md) to lock your app to a tested
commit. Browse the [published docs](https://michaelgiraldo.github.io/chatkit-python/)
for the guides and generated API reference.
The [changelog](docs/release.md) covers branch changes and past releases.
The [upstream status page](docs/upstream-status.md) records review decisions.

## Develop Turnspan

The fork's `main` branch is the maintained Turnspan branch. Its CI workflow
also publishes the documentation after checks pass.

Run `make sync` to install the locked development environment and `make check`
to run formatting, lint, type checks, tests, and the strict docs build. Use
`make test TESTS='tests/test_fastapi.py -k missing_thread'` for a focused run.

Run `make serve-docs` to preview the Zensical site and `make docs-check` to
build it into `site/`. Zensical generates the API reference from `chatkit/`.
Add new public modules to the API navigation in [zensical.toml](zensical.toml).

## License
This project is licensed under the [Apache License 2.0](LICENSE).

# Turnspan for ChatKit

Turnspan is a maintained Python 3.14 fork of OpenAI ChatKit.

**Turnspan 2** reports `openai-chatkit 1.6.5+turnspan.2`. It requires Python
3.14 or newer and uses the existing `chatkit` imports.

Install from the root of the checkout with `python -m pip install .`. See the
[quick start](docs/quickstart.md) to run a ChatKit server and
[Pin an app to the fork](docs/guides/pin-the-fork.md) to install a specific commit.
On the maintained branch, install the `agents` extra to use `chatkit.agents`,
the `fastapi` extra to run the FastAPI adapter with Uvicorn, and the
`templates` extra to load `.widget` files.
The [changelog](docs/release.md) covers branch changes and past releases.
The [upstream status page](docs/upstream-status.md) records review decisions.

## Develop Turnspan

Run `make sync` to install the locked development environment and `make check`
to run formatting, lint, type checks, tests, and the strict docs build. Use
`make test TESTS='tests/test_fastapi.py -k missing_thread'` for a focused run.

Run `make serve-docs` to preview the Zensical site and `make docs-check` to
build it into `site/`. Zensical generates the API reference from `chatkit/`.
Add public modules to the navigation in [zensical.toml](zensical.toml) to
make them easy to find.

## License
This project is licensed under the [Apache License 2.0](LICENSE).

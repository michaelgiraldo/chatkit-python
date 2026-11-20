# Turnspan for ChatKit

Turnspan is a maintained Python 3.14 fork of OpenAI ChatKit.

**Turnspan 2** reports `openai-chatkit 1.6.5+turnspan.2`. It requires Python
3.14 or newer and uses the existing `chatkit` imports.

Install from the root of the checkout with `python -m pip install .`. See the
[quick start](docs/quickstart.md) to run a ChatKit server and
[Pin an app to the fork](docs/guides/pin-the-fork.md) to install a specific commit.
On the maintained branch, install the `agents` extra to use `chatkit.agents`
or the `fastapi` extra to run the FastAPI adapter with Uvicorn.
The [changelog](docs/release.md) covers branch changes and past releases.
The [upstream status page](docs/upstream-status.md) records review decisions.

## Develop Turnspan

Run `make sync` to install the locked development environment and `make check`
to run formatting, lint, type checks, tests, and the strict docs build. Use
`make test TESTS='tests/test_fastapi.py -k missing_thread'` for a focused run.

## License
This project is licensed under the [Apache License 2.0](LICENSE).

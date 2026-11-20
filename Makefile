.PHONY: sync install format format-check lint pyright test build demo docs-check serve-docs deploy-docs check

# Pass the version explicitly because uv --no-config skips .python-version.
UV_ENV := env -i PATH="$(PATH)" HOME="$(HOME)" UV_PYTHON="$(shell cat .python-version)" $(if $(UV_CACHE_DIR),UV_CACHE_DIR="$(UV_CACHE_DIR)")
UV := $(UV_ENV) uv --no-config
UV_LOCK_ARGS := --locked --default-index https://pypi.org/simple
UV_RUN := $(UV) run $(UV_LOCK_ARGS) --all-extras
DEMO_PORT ?= 8000

sync:
	$(UV) sync $(UV_LOCK_ARGS) --all-extras --all-packages --group dev

install: sync

format:
	$(UV_RUN) ruff format
	$(UV_RUN) ruff check --fix

format-check:
	$(UV_RUN) ruff format --check

lint:
	$(UV_RUN) ruff check

pyright:
	$(UV_RUN) pyright

test:
	$(UV_RUN) pytest $(TESTS)

build:
	$(UV) build --default-index https://pypi.org/simple

demo:
	$(UV) run $(UV_LOCK_ARGS) --isolated --no-dev --extra fastapi uvicorn docs.examples.quickstart:app --host 127.0.0.1 --port $(DEMO_PORT)

docs-check:
	$(UV_RUN) mkdocs build --strict

serve-docs:
	$(UV_RUN) mkdocs serve

deploy-docs:
	$(UV_RUN) mkdocs gh-deploy --force --verbose

check: format-check lint pyright test docs-check

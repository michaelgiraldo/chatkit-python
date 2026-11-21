.PHONY: sync install format format-check lint pyright test sdk-check build wheel-check demo docs-check serve-docs check release-check

# Pass the version explicitly because uv --no-config skips .python-version.
UV_ENV := env -i PATH="$(PATH)" HOME="$(HOME)" UV_PYTHON="$(shell cat .python-version)" $(if $(UV_CACHE_DIR),UV_CACHE_DIR="$(UV_CACHE_DIR)")
UV := $(UV_ENV) uv --no-config
UV_LOCK_ARGS := --locked --default-index https://pypi.org/simple
UV_RUN := $(UV) run $(UV_LOCK_ARGS) --all-extras
DOCS_RUN := $(UV) run $(UV_LOCK_ARGS) --isolated --only-group docs
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

sdk-check:
	$(UV) run --no-project python scripts/check-sdk.py "$$OPENAI_VERSION" "$$AGENTS_VERSION"

build:
	$(UV) build --default-index https://pypi.org/simple

wheel-check:
	$(UV_ENV) bash scripts/check-wheel.sh

demo:
	$(UV) run $(UV_LOCK_ARGS) --isolated --no-dev --extra fastapi uvicorn docs.examples.quickstart:app --host 127.0.0.1 --port $(DEMO_PORT)

docs-check:
	$(DOCS_RUN) zensical build --clean --strict

serve-docs:
	$(DOCS_RUN) zensical serve

check: format-check lint pyright test docs-check

release-check:
	$(UV) run --no-project python scripts/check-release.py --sha "$(shell git rev-parse HEAD)"

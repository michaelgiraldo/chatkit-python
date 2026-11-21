#!/usr/bin/env bash
# Build a fresh distribution, then exercise exactly what a wheel consumer imports.
set -euo pipefail

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
uv --no-config build --default-index https://pypi.org/simple --out-dir "$build_dir"
wheels=("$build_dir"/*.whl)
if [[ ${#wheels[@]} != 1 || ! -f ${wheels[0]} ]]; then
    echo "Expected one wheel in the fresh build directory" >&2
    exit 1
fi

check_install() {
    local extra=$1
    shift
    uv --no-config run --no-project --isolated --default-index https://pypi.org/simple \
        --with "${wheels[0]}$extra" python -I "$@"
}

check_install '' - <<'PY'
from importlib.util import find_spec

import chatkit.server
import chatkit.widgets

for dependency in ("agents", "openai", "fastapi", "uvicorn", "jinja2", "chatkit.types"):
    assert find_spec(dependency) is None, dependency
PY
check_install '' tests/check_wheel_types.py .venv/bin/pyright
check_install '[fastapi]' - <<'PY'
from importlib.util import find_spec

import uvicorn
from fastapi import FastAPI

from chatkit.fastapi import process_request

assert callable(process_request)
FastAPI()
for dependency in ("agents", "openai"):
    assert find_spec(dependency) is None, dependency
PY
check_install '[agents]' - <<'PY'
from importlib.util import find_spec

import agents
import openai

from chatkit.agent_input import ThreadItemConverter
from chatkit.agents import stream_agent_response

assert callable(stream_agent_response)
assert callable(ThreadItemConverter().to_agent_input)
assert find_spec("fastapi") is None
PY
check_install '[templates]' - <<'PY'
from importlib.util import find_spec

from chatkit.templates import WidgetTemplate

template = WidgetTemplate({
    "version": "1.0", "name": "Example",
    "template": '{"type":"Card","children":[]}',
})
assert template.build().type == "Card"
for dependency in ("agents", "fastapi"):
    assert find_spec(dependency) is None, dependency
PY

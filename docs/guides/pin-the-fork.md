# Pin an app to the fork

Pin a full GitHub commit SHA to install the same tested ChatKit code whenever
you build your app. This fork and the PyPI release can both report `1.6.5`.
The installed package's source URL identifies the fork commit.

## Lock the fork as an app dependency

From your app's repository, keep `openai-chatkit` in the `dependencies` list in
`pyproject.toml`. For this example, `openai-chatkit==1.6.5` matches the package
version declared at the fork commit. Add this entry to the app's uv sources.

```toml
[tool.uv.sources]
openai-chatkit = { url = "https://github.com/michaelgiraldo/chatkit-python/archive/1c0b0a200de7bc90584e4d63ca23775579bb4dda.tar.gz" }
```

Use the full SHA of the fork commit you tested. Refresh the lock and install it
with Python 3.14.

```sh
uv lock --python 3.14
uv sync --locked --python 3.14
```

The lockfile records the archive URL and its SHA-256 hash. Commit both
`pyproject.toml` and `uv.lock` in the app repository. Use `uv sync --locked` in
your build so it installs the reviewed dependency graph.

## Verify the installed build

Run this from the app directory with the interpreter that runs the app. If
checking a container image, run the Python snippet with the app's interpreter
inside that image.

```sh
uv run --locked --no-sync python - <<'PY'
import json
from importlib.metadata import distribution

commit = "1c0b0a200de7bc90584e4d63ca23775579bb4dda"
expected = f"https://github.com/michaelgiraldo/chatkit-python/archive/{commit}.tar.gz"
package = distribution("openai-chatkit")
metadata = package.read_text("direct_url.json")
if metadata is None:
    raise SystemExit("The installed package has no recorded direct source URL.")

installed = json.loads(metadata)["url"]
if installed != expected:
    raise SystemExit(f"Expected {expected}, found {installed}.")

print(f"openai-chatkit {package.version} installed from {installed}")
PY
```

To roll back, restore the previous app manifest and lockfile, then rebuild
using that lockfile.

[uv URL dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#url) ·
[Python direct URL metadata](https://packaging.python.org/en/latest/specifications/direct-url-data-structure/)

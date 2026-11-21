# Pin an app to the fork

Pin a full GitHub commit SHA to install the same tested Turnspan code whenever
you build your app. The installed package's source URL identifies the commit.

## Lock the fork as an app dependency

Choose a version from the [published releases](https://github.com/michaelgiraldo/chatkit-python/releases)
or a reviewed commit on the maintained branch. Open the commit on GitHub and
copy its full SHA. In your app's `pyproject.toml`, keep `openai-chatkit` in
the `dependencies` list. If you specify an exact version there, use the
version declared by the chosen commit. Add this uv source, replacing
`COMMIT_SHA` with the full SHA.

```toml
[tool.uv.sources]
openai-chatkit = { url = "https://github.com/michaelgiraldo/chatkit-python/archive/COMMIT_SHA.tar.gz" }
```

Choose extras for the pinned version. On the maintained branch, add `agents`
for `chatkit.agents`, `fastapi` for the FastAPI adapter, and `templates` for
loading exported `.widget` files. For example, use `openai-chatkit[agents,fastapi]`
for an Agents integration served by FastAPI.

Refresh the lock and install it with Python 3.14.

```sh
uv lock --python 3.14
uv sync --locked --python 3.14
```

The lockfile records the archive URL and its SHA-256 hash. Commit both
`pyproject.toml` and `uv.lock` in the app repository. Use `uv sync --locked` in
your build so it installs the reviewed dependency graph.

## Verify the installed build

Replace `COMMIT_SHA` with the same SHA and run this from the app directory with
the interpreter that runs the app.

```sh
uv run --locked --no-sync python - <<'PY'
import json
from importlib.metadata import distribution

commit = "COMMIT_SHA"
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

## Verify what serves live traffic

1. Follow the live route or alias to each serving revision. Record the
   resolved artifact, such as the image digest.
2. Run the verification snippet with the app's Python interpreter inside
   that exact artifact. You can inspect a matching container image offline
   without starting the service. Compare the installed version and recorded
   source URL with the release you intended to deploy. If the package was
   installed from a wheel or an index, look for the source commit in the
   build provenance tied to the resolved artifact.
3. Record the revision, artifact digest, package version, and source commit
   together. A lockfile records the intended build input. An image tag can
   move. The resolved artifact and its installed metadata establish what is
   serving traffic.

To roll back, switch traffic to the previous verified artifact if your
platform supports it. For a rebuild, restore the previous manifest and
lockfile. Verify the serving revision again after switching traffic. For
the Turnspan 6 browser callback change, roll back the browser bundle and
store implementation with the server.
[Callback upgrade and rollback](update-client-during-response.md#upgrade-from-older-turnspan-versions)

[uv URL dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#url) ·
[Python direct URL metadata](https://packaging.python.org/en/latest/specifications/direct-url-data-structure/)

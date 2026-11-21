"""Check the package a Turnspan consumer would install and prepare release notes."""

import argparse
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import quote, urlsplit

GITHUB_REPO = "https://github.com/michaelgiraldo/chatkit-python"
ROOT = Path(__file__).resolve().parents[1]
SMOKE_TESTS = (
    "tests/test_quickstart.py",
    "tests/test_fastapi.py",
    "tests/test_openai_sdk_stream.py",
)
CONSUMER_CHECK = """
import json
import platform
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import pytest

expected_version, expected_url, *tests = sys.argv[1:]
package = distribution("openai-chatkit")
direct_url = json.loads(package.read_text("direct_url.json") or "{}")
if package.version != expected_version or direct_url.get("url") != expected_url:
    raise SystemExit(
        f"Unexpected Turnspan install: {package.version} from {direct_url.get('url')}"
    )
Path("consumer-versions.json").write_text(json.dumps({
    "python": platform.python_version(),
    "openai": version("openai"),
    "openai-agents": version("openai-agents"),
}), encoding="utf-8")
raise SystemExit(pytest.main([
    "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
    "-p", "pytest_asyncio.plugin", "-o", "asyncio_mode=auto",
    "-o", "asyncio_default_fixture_loop_scope=function", *tests
]))
"""


def release_section(source: Path, version: str, sha: str) -> str:
    changelog = (source / "docs/release.md").read_text(encoding="utf-8")
    heading = rf"^## Turnspan \d+ \(`{re.escape(version)}`\)\n"
    matches = list(re.finditer(heading, changelog, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"Expected one changelog section for {version}")
    content = re.split(
        r"^## ", changelog[matches[0].end() :], maxsplit=1, flags=re.MULTILINE
    )[0].strip()
    if not content:
        raise ValueError(f"The changelog section for {version} is empty")

    def pin_link(match: re.Match[str]) -> str:
        link = urlsplit(match[1])
        path = posixpath.normpath(posixpath.join("docs", link.path))
        target = f"{GITHUB_REPO}/blob/{sha}/{quote(path, safe='/')}"
        if link.fragment:
            target += f"#{quote(link.fragment)}"
        return f"]({target})"

    return re.sub(r"\]\((?![\w+.-]+:|[/#])([^\s)]+)\)", pin_link, content)


def check_consumer(
    source: Path, scratch: Path, url: str, version: str, locked: dict[str, str]
) -> dict[str, str]:
    consumer = scratch / "consumer"
    consumer.mkdir()
    shutil.copytree(source / "tests", consumer / "tests")
    shutil.copytree(source / "docs/examples", consumer / "docs/examples")
    env = {
        name: os.environ[name]
        for name in ("HOME", "PATH", "TMPDIR", "UV_CACHE_DIR")
        if name in os.environ
    }
    env.update(
        OPENAI_API_KEY="turnspan-release-test-unusable",
        OPENAI_BASE_URL="http://127.0.0.1:9",
        OPENAI_AGENTS_DISABLE_TRACING="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    subprocess.run(
        [
            "uv",
            "--no-config",
            "run",
            "--no-project",
            "--isolated",
            "--python",
            sys.executable,
            "--default-index",
            "https://pypi.org/simple",
            "--with",
            f"openai-chatkit[agents,fastapi] @ {url}",
            "--with",
            f"pytest=={locked['pytest']}",
            "--with",
            f"pytest-asyncio=={locked['pytest-asyncio']}",
            "python",
            "-c",
            CONSUMER_CHECK,
            version,
            url,
            *SMOKE_TESTS,
        ],
        check=True,
        cwd=consumer,
        env=env,
        stdout=sys.stderr,
    )
    return json.loads((consumer / "consumer-versions.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True, help="Full candidate commit SHA")
    parser.add_argument("--github-archive", action="store_true")
    parser.add_argument("--ci-url", help="Successful CI run verified by the workflow")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("--sha must be a full lowercase commit SHA")
    if args.ci_url and not re.fullmatch(
        rf"{re.escape(GITHUB_REPO)}/actions/runs/[0-9]+", args.ci_url
    ):
        parser.error("--ci-url must name a Turnspan Actions run")
    if sys.version_info[:2] != (3, 14):
        parser.error("Run the check with Python 3.14")
    sha = args.sha
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if commit != sha:
        parser.error("--sha must identify a commit rather than an annotated tag")

    public_archive = f"{GITHUB_REPO}/archive/{sha}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="turnspan-release-") as directory:
        scratch = Path(directory)
        archive = scratch / "turnspan.tar.gz"
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar.gz",
                "--prefix=turnspan/",
                f"--output={archive}",
                sha,
            ],
            cwd=ROOT,
            check=True,
        )
        with tarfile.open(archive) as source_files:
            source_files.extractall(scratch, filter="data")
        source = scratch / "turnspan"
        project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
        version = project["project"]["version"]
        if project["project"]["name"] != "openai-chatkit" or not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+\+turnspan\.[1-9][0-9]*", version
        ):
            parser.error("The source must declare an OpenAI ChatKit Turnspan version")
        lockfile = tomllib.loads((source / "uv.lock").read_text(encoding="utf-8"))
        locked = {
            package["name"]: package["version"] for package in lockfile["package"]
        }
        if locked.get("openai-chatkit") != version:
            parser.error("Update uv.lock to match the release version")
        section = release_section(source, version, sha)
        source_url = public_archive if args.github_archive else archive.as_uri()
        resolved = check_consumer(source, scratch, source_url, version, locked)

    tag = f"v{version}"
    exact_docs = f"{GITHUB_REPO}/tree/{sha}/docs"
    verification = (
        f"[Passing CI run]({args.ci_url})"
        if args.ci_url
        else "GitHub CI is checked by the release workflow."
    )
    notes = f"""Package `openai-chatkit {version}` · Python 3.14 · Tag `{tag}`

Source commit [`{sha}`]({GITHUB_REPO}/commit/{sha}).

{section}

## Install

```sh
python -m pip install 'openai-chatkit[agents,fastapi] @ {public_archive}'
```

[Documentation for this commit]({exact_docs}) · [Pin and verify the installed build]({GITHUB_REPO}/blob/{sha}/docs/guides/pin-the-fork.md)

## Verification

- {verification}
- Locked versions in the SDK repository were OpenAI Python `{locked["openai"]}` and Agents SDK `{locked["openai-agents"]}`.
- The consumer smoke ran on Python `{resolved["python"]}` with OpenAI Python `{resolved["openai"]}` and Agents SDK `{resolved["openai-agents"]}`. It exercised the quickstart, FastAPI, and mocked model HTTP from an isolated source install.
"""
    print(notes)


if __name__ == "__main__":
    main()

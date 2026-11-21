"""Run the full suite against candidate SDK versions in an isolated environment."""

import os
import platform
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path


def run_tests(openai: str, agents: str) -> None:
    # These dependencies are available inside the candidate environment.
    import pytest
    from packaging.version import Version

    print(f"Python {platform.python_version()}", flush=True)
    for name, expected in (("openai", openai), ("openai-agents", agents)):
        actual = version(name)
        print(f"{name} {actual}", flush=True)
        if Version(actual) != Version(expected):
            sys.exit(f"Expected {name} {expected}, found {actual}")
    sys.exit(pytest.main(["tests"]))


def prepare_environment(openai: str, agents: str) -> None:
    if not (openai or agents):
        sys.exit("Pass OPENAI_VERSION or AGENTS_VERSION to make sdk-check.")
    with Path("uv.lock").open("rb") as lock:
        packages = tomllib.load(lock)["package"]
    locked = {package["name"]: package["version"] for package in packages}
    openai = openai or locked["openai"]
    agents = agents or locked["openai-agents"]

    # Request both SDKs together so uv checks their mutual version requirements.
    command = [
        "uv",
        "--no-config",
        "run",
        "--locked",
        "--isolated",
        "--default-index",
        "https://pypi.org/simple",
        "--all-extras",
        "--group",
        "dev",
        "--with",
        f"openai=={openai}",
        "--with",
        f"openai-agents=={agents}",
        "python",
        __file__,
        "--run-tests",
        openai,
        agents,
    ]
    os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    os.execvp("uv", command)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["--run-tests", openai, agents]:
            run_tests(openai, agents)
        case [openai, agents]:
            prepare_environment(openai, agents)
        case _:
            sys.exit("Use make sdk-check with OPENAI_VERSION or AGENTS_VERSION.")

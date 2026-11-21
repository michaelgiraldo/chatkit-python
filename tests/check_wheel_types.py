"""Check the installed wheel's public imports from a separate project."""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

CORE_IMPORTS = """\
from chatkit.actions import Action, ActionConfig
from chatkit.errors import ErrorCode
from chatkit.events import ThreadItemUpdatedEvent, ThreadStreamEvent
from chatkit.models import ThreadMetadata
from chatkit.protocol import encode_message, encode_sse, parse_request
from chatkit.requests import ChatKitReq
from chatkit.server import ChatKitServer, StreamingResult
from chatkit.store import ThreadPatch, ThreadStore
from chatkit.store_checks import check_client_tool_store, check_thread_store
from chatkit.turns import Turn, TurnOutcome
from chatkit.widget_models import WidgetComponent, WidgetRoot
from chatkit.widget_stream import stream_widget
"""


def check_wheel_types(pyright: Path) -> None:
    with TemporaryDirectory() as project_dir:
        project = Path(project_dir)
        (project / "consumer.py").write_text(CORE_IMPORTS, encoding="utf-8")
        (project / "pyrightconfig.json").write_text(
            json.dumps({
                "pythonVersion": "3.14",
                "reportPrivateImportUsage": "error",
            }),
            encoding="utf-8",
        )
        subprocess.run(
            [str(pyright.resolve()), "--pythonpath", sys.executable, "consumer.py"],
            cwd=project,
            check=True,
        )


if __name__ == "__main__":
    check_wheel_types(Path(sys.argv[1]))

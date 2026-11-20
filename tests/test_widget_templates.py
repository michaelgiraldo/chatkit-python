import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from chatkit.models import WidgetItem
from chatkit.templates import WidgetTemplate
from chatkit.widget_models import WidgetRoot

_WIDGET_ASSETS = Path(__file__).with_name("assets") / "widgets"


@pytest.mark.parametrize("path_type", [str, Path])
def test_relative_widget_paths_use_working_directory(tmp_path, monkeypatch, path_type):
    path = tmp_path / "card.widget"
    path.write_text(
        json.dumps({
            "version": "1.0",
            "name": "Greeting",
            "template": '{"type":"Card","children":[{"type":"Text","value":"{{ greeting }}"}]}',
        }),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    widget = WidgetTemplate.from_file(path_type(path.name)).build({"greeting": "Hello"})

    assert widget.model_dump(mode="json") == {
        "type": "Card",
        "children": [{"type": "Text", "value": "Hello"}],
    }


@pytest.mark.parametrize(
    "widget_name, data",
    [
        ("list_view_no_data", None),
        ("card_no_data", None),
        (
            "list_view_with_data",
            {
                "items": [
                    {
                        "id": "blue",
                        "label": "Blue line",
                        "color": "blue-500",
                    },
                    {
                        "id": "orange",
                        "label": "Orange line",
                        "color": "orange-500",
                    },
                    {
                        "id": "purple",
                        "label": "Purple line",
                        "color": "purple-500",
                    },
                ],
            },
        ),
        (
            "card_with_data",
            {
                "channel": "#proj-chatkit",
                "time": "4:48 PM",
                "user": {
                    "image": "/pam.png",
                    "name": "Pam Beesly",
                },
            },
        ),
    ],
)
def test_widget_template_from_file(
    widget_name: str,
    data: dict[str, Any] | None,
):
    template = WidgetTemplate.from_file(_WIDGET_ASSETS / f"{widget_name}.widget")

    expected_widget_dict = json.loads(
        (_WIDGET_ASSETS / f"{widget_name}.json").read_text(encoding="utf-8")
    )

    widget = template.build(data)

    assert isinstance(widget, WidgetRoot)
    assert widget.model_dump(exclude_none=True) == expected_widget_dict


def test_widget_template_build_with_basic_root():
    template = WidgetTemplate.from_file(_WIDGET_ASSETS / "basic_root.widget")

    expected_widget_dict = json.loads(
        (_WIDGET_ASSETS / "basic_root.json").read_text(encoding="utf-8")
    )

    widget = template.build(
        {
            "name": "Harry Potter",
            "bio": "The boy who lived",
        },
    )

    assert isinstance(widget, WidgetRoot)
    assert widget.type == "Basic"
    assert widget.model_dump(exclude_none=True) == expected_widget_dict

    widget_item = WidgetItem(
        thread_id="1", widget=widget, id="1", created_at=datetime.now(UTC)
    )
    assert widget_item.widget.type == "Basic"

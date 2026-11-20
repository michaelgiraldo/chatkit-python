from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from chatkit.events import (
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    WidgetRootUpdated,
    WidgetStreamingTextValueDelta,
)
from chatkit.models import ThreadMetadata, WidgetItem
from chatkit.widget_models import WidgetComponent, WidgetRoot
from chatkit.widget_stream import diff_widget, stream_widget
from chatkit.widgets import (
    Card,
    Markdown,
    Text,
    WidgetRootInput,
)


@pytest.mark.parametrize("component", [Text, Markdown])
def test_appended_text_without_id_replaces_widget(component):
    before = Card(children=[component(value="Hello")])
    after = Card(children=[component(value="Hello, world!")])

    updates = diff_widget(before, after)

    assert len(updates) == 1
    assert updates[0].type == "widget.root.updated"
    assert updates[0].widget == after.build()


@pytest.mark.parametrize("component_type", ["Text", "Markdown"])
@pytest.mark.parametrize("component_id", [None, ""])
def test_dynamic_appended_text_without_id_replaces_widget(component_type, component_id):
    before = WidgetRoot(
        type="Card",
        children=[
            WidgetComponent.model_validate({
                "type": component_type,
                "id": component_id,
                "value": "Hello",
            })
        ],
    )
    after = WidgetRoot(
        type="Card",
        children=[
            WidgetComponent.model_validate({
                "type": component_type,
                "id": component_id,
                "value": "Hello, world!",
            })
        ],
    )

    updates = diff_widget(before, after)

    assert len(updates) == 1
    assert updates[0].type == "widget.root.updated"
    assert updates[0].widget == after


@pytest.mark.parametrize(
    "before, after, expected",
    [
        (Card(children=[]), Card(children=[]), []),
        (
            Card(children=[Text(id="text", value="Hello", streaming=True)]),
            Card(children=[Text(id="text", value="Hello, world!", streaming=True)]),
            ["widget.streaming_text.value_delta"],
        ),
        (
            Card(children=[Text(id="text", value="Hello", streaming=True)]),
            Card(children=[Text(id="text", value="Hello, world!", streaming=False)]),
            ["widget.root.updated"],
        ),
        (
            Card(children=[Text(value="Hello")]),
            Card(children=[Text(value="world!")]),
            ["widget.root.updated"],
        ),
        (
            WidgetRoot(type="Card", children=[]),
            WidgetRoot(type="Card", children=[]),
            [],
        ),
        (
            WidgetRoot(
                type="Card",
                children=[
                    WidgetComponent.model_validate({
                        "type": "Text",
                        "id": "text",
                        "value": "Hello",
                        "streaming": True,
                    })
                ],
            ),
            WidgetRoot(
                type="Card",
                children=[
                    WidgetComponent.model_validate({
                        "type": "Text",
                        "id": "text",
                        "value": "Hello, world!",
                        "streaming": True,
                    })
                ],
            ),
            ["widget.streaming_text.value_delta"],
        ),
        (
            WidgetRoot(
                type="Card",
                children=[
                    WidgetComponent.model_validate({
                        "type": "Text",
                        "id": "text",
                        "value": "Hello",
                        "streaming": True,
                    })
                ],
            ),
            WidgetRoot(
                type="Card",
                children=[
                    WidgetComponent.model_validate({
                        "type": "Text",
                        "id": "text",
                        "value": "Hello, world!",
                        "streaming": False,
                    })
                ],
            ),
            ["widget.root.updated"],
        ),
        (
            WidgetRoot(
                type="Card",
                children=[
                    WidgetComponent.model_validate({
                        "type": "Text",
                        "value": "Hello",
                    })
                ],
            ),
            WidgetRoot(
                type="Card",
                children=[
                    WidgetComponent.model_validate({
                        "type": "Text",
                        "value": "world!",
                    })
                ],
            ),
            ["widget.root.updated"],
        ),
    ],
)
def test_diff(
    before: WidgetRootInput,
    after: WidgetRootInput,
    expected: list[
        Literal[
            "widget.streaming_text.value_delta",
            "widget.root.updated",
        ]
    ],
):
    diff = diff_widget(before, after)
    assert len(diff) == len(expected)
    for i in range(len(diff)):
        assert diff[i].type == expected[i]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "before, after",
    [
        (
            {"type": "Card", "children": []},
            {"type": "Card", "children": [], "size": "sm"},
        ),
        (
            {"type": "Card", "children": [{"type": "Text", "value": "Hello"}]},
            {
                "type": "Card",
                "children": [{"type": "Text", "value": "Hello", "color": "secondary"}],
            },
        ),
    ],
)
def test_diff_replaces_dynamic_widgets_when_properties_are_added_or_removed(
    before: dict[str, Any], after: dict[str, Any], reverse: bool
):
    before_widget = WidgetRoot.model_validate(before)
    after_widget = WidgetRoot.model_validate(after)
    if reverse:
        before_widget, after_widget = after_widget, before_widget

    assert diff_widget(before_widget, after_widget) == [
        WidgetRootUpdated(widget=after_widget)
    ]


def test_diff_streams_text_under_a_single_child():
    before = Card.model_validate({
        "children": [
            {
                "type": "Transition",
                "children": {"type": "Text", "id": "status", "value": "Loading"},
            }
        ]
    })
    after = Card.model_validate({
        "children": [
            {
                "type": "Transition",
                "children": {"type": "Text", "id": "status", "value": "Loading data"},
            }
        ]
    })

    assert diff_widget(before, after) == [
        WidgetStreamingTextValueDelta(component_id="status", delta=" data", done=True)
    ]


def test_diff_replaces_widget_when_text_id_is_ambiguous():
    before = Card(children=[Text(id="status", value="A"), Text(id="status", value="B")])
    after = Card(
        children=[Text(id="status", value="AA"), Text(id="status", value="BB")]
    )

    assert diff_widget(before, after) == [WidgetRootUpdated(widget=after.build())]


async def test_stream_widget_snapshots_a_reused_widget_instance():
    thread = ThreadMetadata(id="thread", created_at=datetime.now(UTC))
    widget = WidgetRoot.model_validate({
        "type": "Card",
        "children": [{"type": "Text", "id": "status", "value": "Loading"}],
    })
    assert isinstance(widget.children, list)
    (text,) = widget.children

    async def states() -> AsyncGenerator[WidgetRoot]:
        yield widget
        assert text.model_extra is not None
        text.model_extra["value"] = "Loading data"
        yield widget

    started, changed, finished = [
        event async for event in stream_widget(thread, states())
    ]
    assert isinstance(started, ThreadItemAddedEvent)
    assert isinstance(started.item, WidgetItem)
    assert started.item.widget.model_dump()["children"][0]["value"] == "Loading"
    assert isinstance(changed, ThreadItemUpdatedEvent)
    assert changed.update == WidgetStreamingTextValueDelta(
        component_id="status", delta=" data", done=True
    )
    assert isinstance(finished, ThreadItemDoneEvent)
    assert isinstance(finished.item, WidgetItem)
    assert finished.item.widget.model_dump()["children"][0]["value"] == "Loading data"

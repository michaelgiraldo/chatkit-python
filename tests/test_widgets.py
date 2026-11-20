import json
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import Any, Literal

import pytest

from chatkit.actions import ActionConfig
from chatkit.server import diff_widget, stream_widget
from chatkit.turns import Turn
from chatkit.types import (
    InferenceOptions,
    Thread,
    ThreadCreatedEvent,
    ThreadCreateParams,
    ThreadGetByIdParams,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    ThreadsCreateReq,
    ThreadsGetByIdReq,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageTextContent,
    WidgetItem,
    WidgetRootUpdated,
    WidgetStreamingTextValueDelta,
)
from chatkit.widgets import (
    Card,
    ListView,
    ListViewItem,
    Markdown,
    Text,
    WidgetComponent,
    WidgetRoot,
    WidgetRootInput,
    WidgetTemplate,
)
from tests._types import RequestContext
from tests.helpers.server import make_server


def test_typed_builders_keep_nested_list_view_and_click_actions():
    action = ActionConfig(type="line.select", payload={"value": "blue"})
    named_row = ListViewItem(
        id="blue_row", onClickAction=action, children=[Text(value="Blue line")]
    )
    assert named_row.onClickAction == action
    card = Card(children=[ListView(id="nested_list", children=[named_row])]).build()

    assert isinstance(card.children, list)
    (child,) = card.children
    assert child.type == "ListView"
    assert child.id == "nested_list"
    assert isinstance(child.children, list)
    (row,) = child.children
    assert row.type == "ListViewItem"
    assert row.id == "blue_row"
    assert row.model_dump()["onClickAction"] == action.model_dump()
    assert row.model_dump()["children"] == [{"type": "Text", "value": "Blue line"}]
    assert WidgetRoot.model_validate_json(card.model_dump_json()) == card


@pytest.mark.parametrize(
    "border",
    [
        1.5,
        {"size": 0.5, "color": {"light": "#ccc", "dark": "#333"}, "style": "dashed"},
        {"top": 0.5, "bottom": {"size": 1.5, "color": "strong"}},
    ],
)
async def test_widget_tree_survives_thread_reload(border):
    payload = {
        "type": "Card",
        "border": border,
        "extension": {"positions": ["start", None, "end"]},
        "children": [
            {
                "type": "ListView",
                "children": [
                    {
                        "type": "ListViewItem",
                        "onClickAction": {
                            "type": "line.select",
                            "payload": {"selection": None},
                        },
                        "children": [
                            {"type": "Text", "value": "Blue line", "futureProp": None}
                        ],
                    }
                ],
            }
        ],
    }
    template = WidgetTemplate({
        "version": "1.0",
        "name": "Bordered card",
        "template": json.dumps(payload),
    })

    async def respond(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        yield ThreadItemDoneEvent(
            item=WidgetItem(
                id="bordered-card",
                thread_id=turn.thread.id,
                created_at=datetime.now(),
                widget=template.build(),
            )
        )

    with make_server(respond) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Show a card")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread_id = next(
            event.thread.id for event in events if isinstance(event, ThreadCreatedEvent)
        )
        (streamed_item,) = [
            event.item
            for event in events
            if isinstance(event, ThreadItemDoneEvent)
            and isinstance(event.item, WidgetItem)
        ]
        thread = await server.process_non_streaming(
            ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread_id))
        )
        assert isinstance(thread, Thread)

    (stored_item,) = [
        item for item in thread.items.data if isinstance(item, WidgetItem)
    ]
    for item in (streamed_item, stored_item):
        assert type(item.widget) is WidgetRoot
        assert item.widget.model_dump(mode="json") == payload
        assert json.loads(item.model_dump_json(exclude_none=True))["widget"] == payload


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
    thread = ThreadMetadata(id="thread", created_at=datetime.now())
    widget = WidgetRoot.model_validate({
        "type": "Card",
        "children": [{"type": "Text", "id": "status", "value": "Loading"}],
    })
    assert isinstance(widget.children, list)
    (text,) = widget.children

    async def states() -> AsyncGenerator[WidgetRoot, None]:
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


def test_json_dump_excludes_none_fields():
    widget = Card(children=[Text(value="Hello")])

    json_str = widget.model_dump_json()
    assert isinstance(json_str, str)
    data = json.loads(json_str)

    # Top-level widget should include type and exclude None-valued fields.
    assert data["type"] == "Card"
    assert "key" not in data
    assert "padding" not in data
    assert "status" not in data
    assert "collapsed" not in data

    # Children should be serialized with None fields omitted as well.
    assert isinstance(data["children"], list)
    assert len(data["children"]) == 1

    text_dump = data["children"][0]
    assert text_dump["type"] == "Text"
    assert text_dump["value"] == "Hello"
    assert "italic" not in text_dump
    assert "streaming" not in text_dump
    assert "color" not in text_dump
    assert "key" not in text_dump


def test_json_dump_excludes_none_fields_nested():
    widget = Card(children=[Text(value="Hello")])
    widget_item = WidgetItem(
        thread_id="1", widget=widget.build(), id="1", created_at=datetime.now()
    )

    json_str = widget_item.model_dump_json()
    assert isinstance(json_str, str)
    data = json.loads(json_str)

    # Top-level widget should include type and exclude None-valued fields.
    widget_dump = data["widget"]
    assert widget_dump["type"] == "Card"
    assert "key" not in widget_dump
    assert "padding" not in widget_dump
    assert "status" not in widget_dump
    assert "collapsed" not in widget_dump

    # Children should be serialized with None fields omitted as well.
    assert isinstance(widget_dump["children"], list)
    assert len(widget_dump["children"]) == 1

    text_dump = widget_dump["children"][0]
    assert text_dump["type"] == "Text"
    assert text_dump["value"] == "Hello"
    assert "italic" not in text_dump
    assert "streaming" not in text_dump
    assert "color" not in text_dump
    assert "key" not in text_dump


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
    template = WidgetTemplate.from_file(f"assets/widgets/{widget_name}.widget")

    with open(f"tests/assets/widgets/{widget_name}.json", "r") as file:
        expected_widget_dict = json.load(file)

    widget = template.build(data)

    assert isinstance(widget, WidgetRoot)
    assert widget.model_dump(exclude_none=True) == expected_widget_dict


def test_widget_template_build_with_basic_root():
    template = WidgetTemplate.from_file("assets/widgets/basic_root.widget")

    with open("tests/assets/widgets/basic_root.json", "r") as file:
        expected_widget_dict = json.load(file)

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
        thread_id="1", widget=widget, id="1", created_at=datetime.now()
    )
    assert widget_item.widget.type == "Basic"

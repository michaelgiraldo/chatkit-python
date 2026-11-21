import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from chatkit.actions import ActionConfig
from chatkit.events import ThreadCreatedEvent, ThreadItemDoneEvent, ThreadStreamEvent
from chatkit.models import (
    InferenceOptions,
    Thread,
    UserMessageInput,
    UserMessageTextContent,
    WidgetItem,
)
from chatkit.requests import (
    ThreadCreateParams,
    ThreadGetByIdParams,
    ThreadsCreateReq,
    ThreadsGetByIdReq,
)
from chatkit.templates import WidgetTemplate
from chatkit.turns import Turn
from chatkit.widget_models import WidgetRoot
from chatkit.widgets import (
    AreaSeries,
    BarSeries,
    Card,
    Chart,
    LineSeries,
    ListView,
    ListViewItem,
    Text,
)
from tests._types import RequestContext
from tests.helpers.server import make_server


def test_chart_series_round_trip_through_a_card():
    series = [
        {"type": "bar", "label": "Bars", "dataKey": "value", "stack": "group"},
        {"type": "area", "label": "Area", "dataKey": "value", "curveType": "linear"},
        {
            "type": "line",
            "label": "Line",
            "dataKey": "value",
            "color": {"light": "blue", "dark": "purple"},
        },
    ]
    chart = Chart.model_validate({
        "data": [{"value": 2}],
        "xAxis": "value",
        "series": series,
    })
    assert [type(item) for item in chart.series] == [BarSeries, AreaSeries, LineSeries]
    (node,) = Card(children=[chart]).build().model_dump(mode="json")["children"]
    assert node["series"] == series


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
        thread_id="1", widget=widget.build(), id="1", created_at=datetime.now(UTC)
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

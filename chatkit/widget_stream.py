"""Widget state differences and asynchronous widget streaming."""

import json
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import Any

from . import events, models, store, widget_models, widgets
from ._stream import _closing_stream


def _widget_state(widget: widgets.WidgetRootInput) -> widget_models.WidgetRoot:
    return widget if isinstance(widget, widget_models.WidgetRoot) else widget.build()


def diff_widget(
    before: widgets.WidgetRootInput, after: widgets.WidgetRootInput
) -> list[events.WidgetStreamingTextValueDelta | events.WidgetRootUpdated]:
    """Emit unambiguous text appends, or replace the widget for other changes."""
    before_payload = _widget_state(before).model_dump(mode="json")
    after_widget = _widget_state(after)
    after_payload = after_widget.model_dump(mode="json")

    def index_components(root: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        by_id: dict[str, list[dict[str, Any]]] = {}

        def visit(node: dict[str, Any]) -> None:
            if component_id := node.get("id"):
                by_id.setdefault(component_id, []).append(node)
            children = node.get("children")
            if isinstance(children, dict):
                visit(children)
            elif isinstance(children, list):
                for child in children:
                    visit(child)

        visit(root)
        return by_id

    before_index = index_components(before_payload)
    deltas: list[events.WidgetStreamingTextValueDelta | events.WidgetRootUpdated] = []
    for component_id, new_nodes in index_components(after_payload).items():
        old_nodes = before_index.get(component_id, [])
        if len(old_nodes) != 1 or len(new_nodes) != 1:
            continue
        old_node, new_node = old_nodes[0], new_nodes[0]
        old_value, new_value = old_node.get("value"), new_node.get("value")
        if (
            old_node["type"] in {"Text", "Markdown"}
            and old_node["type"] == new_node["type"]
            and isinstance(old_value, str)
            and isinstance(new_value, str)
            and new_value != old_value
            and new_value.startswith(old_value)
        ):
            old_node["value"] = new_value
            deltas.append(
                events.WidgetStreamingTextValueDelta(
                    component_id=component_id,
                    delta=new_value[len(old_value) :],
                    done=not new_node.get("streaming", False),
                )
            )

    # Once appends are applied, compare the remaining JSON, including extensions
    # and explicit nulls. JSON equality distinguishes boolean and numeric props.
    if json.dumps(before_payload, sort_keys=True) != json.dumps(
        after_payload, sort_keys=True
    ):
        return [events.WidgetRootUpdated(widget=after_widget)]
    return deltas


async def stream_widget(
    thread: models.ThreadMetadata,
    widget: widgets.WidgetRootInput | AsyncGenerator[widgets.WidgetRootInput],
    copy_text: str | None = None,
    generate_id: Callable[[store.StoreItemType], str] = store.default_generate_id,
) -> AsyncGenerator[events.ThreadStreamEvent]:
    """Stream a widget root or successive root states.

    An empty producer emits no events. Closing the returned stream also closes
    its producer.
    """

    if not isinstance(widget, AsyncGenerator):
        yield events.ThreadItemDoneEvent(
            item=models.WidgetItem(
                id=generate_id("message"),
                thread_id=thread.id,
                created_at=datetime.now(UTC),
                widget=_widget_state(widget),
                copy_text=copy_text,
            ),
        )
        return

    async with _closing_stream(widget) as states:
        try:
            last_state = _widget_state(await anext(states)).model_copy(deep=True)
        except StopAsyncIteration:
            return

        item = models.WidgetItem(
            id=generate_id("message"),
            created_at=datetime.now(UTC),
            widget=last_state,
            copy_text=copy_text,
            thread_id=thread.id,
        )
        yield events.ThreadItemAddedEvent(item=item)

        async for state in states:
            new_state = _widget_state(state).model_copy(deep=True)
            for update in diff_widget(last_state, new_state):
                yield events.ThreadItemUpdatedEvent(
                    item_id=item.id,
                    update=update,
                )
            last_state = new_state

    yield events.ThreadItemDoneEvent(
        item=item.model_copy(update={"widget": last_state}),
    )

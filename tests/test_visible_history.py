from datetime import UTC, datetime

import pytest

from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    HiddenContextItem,
    Page,
    SDKHiddenContextItem,
    Thread,
    ThreadMetadata,
)
from chatkit.requests import (
    ItemsListParams,
    ItemsListReq,
    ThreadGetByIdParams,
    ThreadsGetByIdReq,
)
from docs.examples.in_memory_thread_store import InMemoryThreadStore
from tests.helpers.server import DEFAULT_CONTEXT, make_server


@pytest.mark.parametrize("order", ["asc", "desc"])
async def test_client_history_pages_count_only_visible_items(order):
    thread = ThreadMetadata(id="example", created_at=datetime.now(UTC))
    with make_server() as server:
        await server.store.save_thread(thread, DEFAULT_CONTEXT)
        for index in range(21):
            await server.store.add_thread_item(
                thread.id,
                HiddenContextItem(
                    id=f"a-hidden-{index:02}",
                    thread_id=thread.id,
                    created_at=thread.created_at,
                    content="internal",
                ),
                DEFAULT_CONTEXT,
            )
        for item_id in ("visible-one", "visible-two"):
            await server.store.add_thread_item(
                thread.id,
                AssistantMessageItem(
                    id=item_id,
                    thread_id=thread.id,
                    created_at=thread.created_at,
                    content=[AssistantMessageContent(text="Visible reply")],
                ),
                DEFAULT_CONTEXT,
            )
        await server.store.add_thread_item(
            thread.id,
            SDKHiddenContextItem(
                id="z-hidden",
                thread_id=thread.id,
                created_at=thread.created_at,
                content="internal",
            ),
            DEFAULT_CONTEXT,
        )

        first, second = (
            ("visible-one", "visible-two")
            if order == "asc"
            else ("visible-two", "visible-one")
        )
        request = ItemsListReq(
            params=ItemsListParams(thread_id=thread.id, order=order, limit=1)
        )
        page = await server.process(request, DEFAULT_CONTEXT)
        assert isinstance(page, Page)
        assert [item.id for item in page.data] == [first]
        assert page.has_more
        assert page.after == first

        request.params.after = page.after
        last = await server.process(request, DEFAULT_CONTEXT)
        assert isinstance(last, Page)
        assert [item.id for item in last.data] == [second]
        assert last.has_more is False
        assert last.after is None

        loaded = await server.process(
            ThreadsGetByIdReq(params=ThreadGetByIdParams(thread_id=thread.id)),
            DEFAULT_CONTEXT,
        )
        assert isinstance(loaded, Thread)
        assert [item.id for item in loaded.items.data] == ["visible-one", "visible-two"]
        assert not loaded.items.has_more

        raw = await server.store.load_thread_items(
            thread.id, None, 30, "asc", DEFAULT_CONTEXT
        )
        assert len(raw.data) == 24


async def test_in_memory_store_orders_legacy_and_utc_timestamps_together():
    store = InMemoryThreadStore()
    older = ThreadMetadata(id="older", created_at=datetime(2025, 1, 1))
    newer = ThreadMetadata(id="newer", created_at=datetime(2026, 1, 1, tzinfo=UTC))
    await store.save_thread(newer, {})
    await store.save_thread(older, {})
    page = await store.load_threads(10, None, "asc", {})
    assert page.data == [older, newer]

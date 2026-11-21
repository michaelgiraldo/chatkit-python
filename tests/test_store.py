import asyncio
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest

from chatkit.models import (
    AssistantMessageItem,
    ClientToolCallItem,
    InferenceOptions,
    StructuredInputItem,
    ThreadMetadata,
    UserMessageItem,
)
from chatkit.store import ConflictError, NotFoundError
from chatkit.store_checks import check_thread_store
from docs.examples.in_memory_thread_store import InMemoryThreadStore
from docs.examples.sqlite_thread_store import ItemData, SQLiteThreadStore
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.storage_contract import (
    ALTERNATIVE_CONTEXT,
    DEFAULT_CONTEXT,
    AttachmentMetadataStoreContract,
    ThreadStoreContract,
    make_thread,
)


class TestSqliteStore(ThreadStoreContract, AttachmentMetadataStoreContract):
    def setup_method(self, method):
        db_path = f"file:{method.__name__}?mode=memory&cache=shared"
        # Keep the shared in-memory database from being deleted when the last connection is closed
        self.db = sqlite3.connect(db_path, uri=True)
        store = SQLiteStore(db_path)
        self.store = store
        self.attachment_metadata_store = store

    def teardown_method(self, method):
        self.db.close()

    @pytest.mark.asyncio
    async def test_default_id_generators_use_default_prefixes(self):
        ctx = DEFAULT_CONTEXT
        thread_id = self.store.generate_thread_id(ctx)
        assert thread_id.startswith("thr_")
        assert len(thread_id.removeprefix("thr_")) == 32

        thread = make_thread(thread_id=thread_id)
        assert self.store.generate_item_id("message", thread, ctx).startswith("msg_")
        assert self.store.generate_item_id("tool_call", thread, ctx).startswith("tc_")
        assert self.store.generate_item_id("task", thread, ctx).startswith("tsk_")
        assert self.store.generate_item_id("workflow", thread, ctx).startswith("wf_")


class TestSqliteStoreCustomIds(ThreadStoreContract, AttachmentMetadataStoreContract):
    def setup_method(self, method):
        db_path = f"file:{method.__name__}_custom?mode=memory&cache=shared"
        # Keep the shared in-memory database from being deleted when the last connection is closed
        self.db = sqlite3.connect(db_path, uri=True)

        class CustomSQLiteStore(SQLiteStore):
            def __init__(self, path: str):
                super().__init__(path)
                self.counter = 0

            def generate_thread_id(self, context) -> str:
                self.counter += 1
                return f"thr_custom_{self.counter}"

            def generate_item_id(
                self, item_type: str, thread: ThreadMetadata, context
            ) -> str:
                self.counter += 1
                return f"{item_type}_custom_{self.counter}_{thread.id}"

        store = CustomSQLiteStore(db_path)
        self.store = store
        self.attachment_metadata_store = store

    def teardown_method(self, method):
        self.db.close()

    @pytest.mark.asyncio
    async def test_overridden_id_generators_are_used(self):
        ctx = DEFAULT_CONTEXT
        thread_id = self.store.generate_thread_id(ctx)
        assert thread_id == "thr_custom_1"

        thread = make_thread(thread_id=thread_id)
        msg_id = self.store.generate_item_id("message", thread, ctx)
        tool_call_id = self.store.generate_item_id("tool_call", thread, ctx)
        task_id = self.store.generate_item_id("task", thread, ctx)

        assert msg_id == "message_custom_2_thr_custom_1"
        assert tool_call_id == "tool_call_custom_3_thr_custom_1"
        assert task_id == "task_custom_4_thr_custom_1"

        thread2 = make_thread(thread_id=self.store.generate_thread_id(ctx))
        assert thread2.id == "thr_custom_5"
        msg_id2 = self.store.generate_item_id("message", thread2, ctx)
        tool_call_id2 = self.store.generate_item_id("tool_call", thread2, ctx)
        task_id2 = self.store.generate_item_id("task", thread2, ctx)

        assert msg_id2 == "message_custom_6_thr_custom_5"
        assert tool_call_id2 == "tool_call_custom_7_thr_custom_5"
        assert task_id2 == "task_custom_8_thr_custom_5"


@pytest.mark.asyncio
async def test_sqlite_claim_and_patch_across_independent_connections(tmp_path: Path):
    db_path = str(tmp_path / "concurrent.sqlite")
    first_store, second_store = SQLiteStore(db_path), SQLiteStore(db_path)
    thread = make_thread()
    ctx = DEFAULT_CONTEXT
    await first_store.save_thread(thread, ctx)
    pending = StructuredInputItem(
        id="form", thread_id=thread.id, created_at=thread.created_at, inputs=[]
    )
    await first_store.add_thread_item(thread.id, pending, ctx)

    barrier = Barrier(2, timeout=5)

    def claim_and_patch(store: SQLiteStore, value: str) -> bool:
        async def update() -> bool:
            replacement = pending.model_copy(update={"status": value})
            try:
                await store.claim_pending_item(thread.id, pending, replacement, ctx)
            except ConflictError:
                claimed = False
            else:
                claimed = True
            await store.patch_thread(thread.id, {"metadata_set": {value: True}}, ctx)
            return claimed

        barrier.wait()
        return asyncio.run(update())

    answered, skipped = await asyncio.gather(
        asyncio.to_thread(claim_and_patch, first_store, "answered"),
        asyncio.to_thread(claim_and_patch, second_store, "skipped"),
    )
    assert answered != skipped
    item = await first_store.load_item(thread.id, pending.id, ctx)
    assert isinstance(item, StructuredInputItem)
    assert item.status == ("answered" if answered else "skipped")
    patched = await first_store.load_thread(thread.id, ctx)
    assert patched.metadata == {"test": "test", "answered": True, "skipped": True}


@pytest.mark.asyncio
async def test_app_store_checks_work_with_independent_instances(tmp_path: Path):
    path = tmp_path / "contract.sqlite3"
    await check_thread_store(
        SQLiteThreadStore[RequestContext](path),
        DEFAULT_CONTEXT,
        other_context=ALTERNATIVE_CONTEXT,
        peer=SQLiteThreadStore[RequestContext](path),
    )


@pytest.mark.asyncio
async def test_file_backed_example_reopens_with_stable_timestamp_order(tmp_path: Path):
    path = tmp_path / "turns.sqlite3"
    thread = make_thread()
    store = SQLiteThreadStore[RequestContext](path)
    await store.save_thread(thread, DEFAULT_CONTEXT)
    instant = datetime.now(UTC)
    for item_id, offset in (("b", timedelta()), ("a", timedelta(hours=-3))):
        await store.add_thread_item(
            thread.id,
            AssistantMessageItem(
                id=item_id,
                thread_id=thread.id,
                created_at=instant.astimezone(timezone(offset)),
                content=[],
            ),
            DEFAULT_CONTEXT,
        )

    reopened = SQLiteThreadStore[RequestContext](path)
    assert await reopened.load_thread(thread.id, DEFAULT_CONTEXT) == thread
    page = await reopened.load_thread_items(thread.id, None, 1, "asc", DEFAULT_CONTEXT)
    assert [item.id for item in page.data] == ["a"]
    second_page = await reopened.load_thread_items(
        thread.id, page.after, 1, "asc", DEFAULT_CONTEXT
    )
    assert [item.id for item in second_page.data] == ["b"]


@pytest.mark.asyncio
async def test_sqlite_client_tool_lifecycle_reopens_and_leaves_legacy_items_readable(
    tmp_path: Path,
):
    path = tmp_path / "callbacks.sqlite3"
    context = DEFAULT_CONTEXT
    thread = make_thread()
    initial = SQLiteThreadStore[RequestContext](path)
    await initial.save_thread(thread, context)
    active = ClientToolCallItem(
        id="active",
        thread_id=thread.id,
        created_at=thread.created_at,
        call_id="call_active",
        name="example",
        arguments={"value": "active"},
    )
    old_pending = active.model_copy(update={"id": "legacy_pending"})
    old_completed = active.model_copy(
        update={"id": "legacy_completed", "status": "completed", "output": "old"}
    )
    with closing(sqlite3.connect(path)) as connection, connection:
        # These transcript rows predate the callback lifecycle and have no binding.
        for item in (old_pending, old_completed):
            connection.execute(
                """INSERT INTO items (id, thread_id, user_id, created_at, data)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    item.id,
                    thread.id,
                    context.user_id,
                    thread.created_at.isoformat(),
                    ItemData(item=item).model_dump_json(),
                ),
            )
    await initial.add_thread_item(thread.id, active, context)
    reopened = SQLiteThreadStore[RequestContext](path)
    assert (
        await reopened.load_client_tool_status(thread.id, active.id, context)
        == "pending"
    )
    with pytest.raises(NotFoundError):
        await reopened.load_client_tool_status(thread.id, old_pending.id, context)
    with pytest.raises(ConflictError):
        await reopened.claim_pending_item(
            thread.id,
            old_pending,
            old_pending.model_copy(update={"status": "completed"}),
            context,
        )
    public = await reopened.load_thread_items(
        thread.id, None, 2, "asc", context, include_hidden=False
    )
    assert {item.id for item in public.data} == {active.id, old_completed.id}
    assert not public.has_more
    assert await reopened.load_item(thread.id, old_pending.id, context) == old_pending

    completed = active.model_copy(update={"status": "completed", "output": "current"})
    await reopened.claim_pending_item(thread.id, active, completed, context)
    assert (
        await initial.load_client_tool_status(thread.id, active.id, context)
        == "completed"
    )
    assert await initial.load_item(thread.id, active.id, context) == completed


@pytest.mark.asyncio
async def test_sqlite_only_one_client_tool_registers_across_connections(tmp_path: Path):
    path = tmp_path / "client-tool-race.sqlite"
    first_store = SQLiteThreadStore[RequestContext](path)
    second_store = SQLiteThreadStore[RequestContext](path)
    thread = make_thread()
    context = DEFAULT_CONTEXT
    await first_store.save_thread(thread, context)
    barrier = Barrier(2, timeout=5)

    def register(store: SQLiteThreadStore[RequestContext], item_id: str) -> bool:
        item = ClientToolCallItem(
            id=item_id,
            thread_id=thread.id,
            created_at=thread.created_at,
            call_id=f"call_{item_id}",
            name="example",
            arguments={},
        )
        barrier.wait()
        try:
            asyncio.run(store.add_thread_item(thread.id, item, context))
        except ConflictError:
            return False
        return True

    results = await asyncio.gather(
        asyncio.to_thread(register, first_store, "first"),
        asyncio.to_thread(register, second_store, "second"),
    )
    assert results.count(True) == 1
    winner = ("first", "second")[results.index(True)]
    page = await second_store.load_thread_items(thread.id, None, 10, "asc", context)
    assert [item.id for item in page.data] == [winner]
    assert (
        await second_store.load_client_tool_status(thread.id, winner, context)
        == "pending"
    )


@pytest.mark.asyncio
async def test_in_memory_save_user_message_cancels_unfinished_callback():
    store = InMemoryThreadStore()
    context: dict[str, object] = {}
    thread = make_thread()
    await store.save_thread(thread, context)
    tool = ClientToolCallItem(
        id="tool",
        thread_id=thread.id,
        created_at=thread.created_at,
        call_id="call_test",
        name="example",
        arguments={},
    )
    await store.add_thread_item(thread.id, tool, context)
    user_message = UserMessageItem(
        id="user",
        thread_id=thread.id,
        created_at=thread.created_at + timedelta(seconds=1),
        content=[],
        inference_options=InferenceOptions(),
    )
    await store.save_item(thread.id, user_message, context)
    assert (
        await store.load_client_tool_status(thread.id, tool.id, context) == "cancelled"
    )
    visible = await store.load_thread_items(
        thread.id, None, 1, "desc", context, include_hidden=False
    )
    assert visible.data == [user_message]
    assert not visible.has_more
    assert await store.load_item(thread.id, tool.id, context) == tool

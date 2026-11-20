import asyncio
import sqlite3
from pathlib import Path
from threading import Barrier

import pytest

from chatkit.store import ConflictError
from chatkit.types import StructuredInputItem, ThreadMetadata
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.storage_contract import (
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

import asyncio
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest

from chatkit.errors import UnsupportedOperationError
from chatkit.models import (
    AssistantMessageItem,
    ClientToolCallItem,
    InferenceOptions,
    StructuredInputItem,
    ThreadMetadata,
    UserMessageItem,
)
from chatkit.store import ConflictError, NotFoundError, ThreadStore
from chatkit.store_checks import check_client_tool_store, check_thread_store
from chatkit.turns import ClientToolResponsesContinuation, TurnOutcome, TurnState
from docs.examples.in_memory_thread_store import InMemoryThreadStore
from docs.examples.sqlite_thread_store import SQLiteThreadStore
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.storage_contract import (
    ALTERNATIVE_CONTEXT,
    DEFAULT_CONTEXT,
    AttachmentMetadataStoreContract,
    ThreadStoreContract,
    make_thread,
    tool_invocation,
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
    store = SQLiteThreadStore[RequestContext](path)
    peer = SQLiteThreadStore[RequestContext](path)
    await check_thread_store(
        store, DEFAULT_CONTEXT, other_context=ALTERNATIVE_CONTEXT, peer=peer
    )
    await check_client_tool_store(
        store, DEFAULT_CONTEXT, other_context=ALTERNATIVE_CONTEXT, peer=peer
    )


async def test_core_store_check_works_without_browser_tools(tmp_path: Path):
    class TextOnlyStore(SQLiteThreadStore[RequestContext]):
        register_client_tool_call = ThreadStore.register_client_tool_call
        load_client_tool_invocation = ThreadStore.load_client_tool_invocation
        load_client_tool_status = ThreadStore.load_client_tool_status
        cancel_client_tool_call = ThreadStore.cancel_client_tool_call

    path = tmp_path / "text.sqlite3"
    store = TextOnlyStore(path)
    peer = TextOnlyStore(path)
    await check_thread_store(
        store, DEFAULT_CONTEXT, other_context=ALTERNATIVE_CONTEXT, peer=peer
    )
    with pytest.raises(UnsupportedOperationError):
        await check_client_tool_store(
            store, DEFAULT_CONTEXT, other_context=ALTERNATIVE_CONTEXT, peer=peer
        )
    assert not (await peer.load_threads(10, None, "asc", DEFAULT_CONTEXT)).data


@pytest.mark.parametrize("broken_order", ["asc", "desc"])
async def test_core_store_check_detects_repeated_visible_page(
    tmp_path: Path, broken_order: str
):
    class RepeatingPageStore(SQLiteThreadStore[RequestContext]):
        async def load_thread_items(
            self, thread_id, after, limit, order, context, *, include_hidden=True
        ):
            if not include_hidden and order == broken_order:
                after = None
            return await super().load_thread_items(
                thread_id, after, limit, order, context, include_hidden=include_hidden
            )

    store = RepeatingPageStore(tmp_path / "repeated-page.sqlite3")
    with pytest.raises(
        AssertionError, match=f"browser-visible {broken_order} history at page 2"
    ):
        await check_thread_store(
            store, DEFAULT_CONTEXT, other_context=ALTERNATIVE_CONTEXT
        )
    assert not (await store.load_threads(10, None, "asc", DEFAULT_CONTEXT)).data


@pytest.fixture(params=["memory", "sqlite"])
def form_store(request, tmp_path: Path):
    if request.param == "memory":
        memory = InMemoryThreadStore()
        return lambda: memory, {}
    path = tmp_path / "forms.sqlite3"
    return lambda: SQLiteThreadStore[RequestContext](path), DEFAULT_CONTEXT


@pytest.mark.parametrize("write", ["add", "upsert_new", "upsert_existing"])
async def test_active_turn_accepts_only_its_own_form(form_store, write):
    open_store, context = form_store
    store = open_store()
    thread = make_thread()
    older = StructuredInputItem(
        id="older", thread_id=thread.id, created_at=thread.created_at, inputs=[]
    )
    current = older.model_copy(update={"id": "current"})
    await store.begin_turn(thread.id, "first", context, create_thread=thread)
    await store.start_turn(thread.id, "first", context)
    await store.add_thread_item(thread.id, older, context, turn_id="first")
    if write == "upsert_existing":
        await store.add_thread_item(thread.id, current, context, turn_id="first")

    await store.begin_turn(thread.id, "second", context)
    await store.start_turn(thread.id, "second", context)
    issue_form = store.add_thread_item if write == "add" else store.save_item
    await issue_form(thread.id, current, context, turn_id="second")

    reader = open_store()
    with pytest.raises(ConflictError):
        await reader.claim_pending_item(
            thread.id,
            older,
            older.model_copy(update={"status": "answered"}),
            context,
            next_turn_id="reply",
        )
    assert await reader.load_item(thread.id, older.id, context) == older
    assert (
        await reader.load_turn_state(thread.id, "second", context) == TurnState.RUNNING
    )
    answer = current.model_copy(update={"status": "answered"})
    assert (
        await reader.claim_pending_item(
            thread.id, current, answer, context, next_turn_id="reply"
        )
        is None
    )
    assert await reader.load_item(thread.id, current.id, context) == answer
    assert (
        await reader.load_turn_state(thread.id, "reply", context) == TurnState.ACCEPTED
    )


async def test_reopening_an_answered_form_requires_an_explicit_owner(form_store):
    open_store, context = form_store
    store = open_store()
    thread = make_thread()
    pending = StructuredInputItem(
        id="form", thread_id=thread.id, created_at=thread.created_at, inputs=[]
    )
    answered = pending.model_copy(update={"status": "answered"})
    await store.begin_turn(thread.id, "owner", context, create_thread=thread)
    await store.start_turn(thread.id, "owner", context)
    await store.add_thread_item(thread.id, pending, context, turn_id="owner")
    await store.claim_pending_item(thread.id, pending, answered, context)

    await store.save_item(thread.id, pending, context)
    with pytest.raises(ConflictError):
        await store.claim_pending_item(thread.id, pending, answered, context)
    await store.save_item(thread.id, pending, context, turn_id="owner")
    await store.claim_pending_item(thread.id, pending, answered, context)
    assert await store.load_item(thread.id, pending.id, context) == answered


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
async def test_sqlite_lifecycle_reopens_with_private_invocation_separate_from_item(
    tmp_path: Path,
):
    path = tmp_path / "callbacks.sqlite3"
    context = DEFAULT_CONTEXT
    thread = make_thread()
    initial = SQLiteThreadStore[RequestContext](path)
    await initial.begin_turn(thread.id, "parent", context, create_thread=thread)
    await initial.start_turn(thread.id, "parent", context)
    active = ClientToolCallItem(
        id="active",
        thread_id=thread.id,
        created_at=thread.created_at,
        name="example",
        arguments={"value": "active"},
    )
    invocation = tool_invocation(active, "parent").model_copy(
        update={
            "model_continuation": ClientToolResponsesContinuation(
                response_id="resp_original",
                items=[
                    {
                        "type": "function_call_output",
                        "call_id": "call_private",
                        "output": "None",
                    }
                ],
            )
        }
    )
    invalid_invocation = invocation.model_copy(
        update={"continuation_turn_id": "prefilled"}
    )
    with pytest.raises(ValueError, match="must be pending"):
        await initial.register_client_tool_call(
            thread.id, active, invalid_invocation, context
        )
    with pytest.raises(ConflictError, match="registration and completion"):
        await initial.add_thread_item(thread.id, active, context, turn_id="parent")
    with pytest.raises(NotFoundError):
        await initial.load_item(thread.id, active.id, context)
    await initial.register_client_tool_call(thread.id, active, invocation, context)
    await initial.finish_turn(thread.id, "parent", TurnOutcome.WAITING, context)
    with closing(sqlite3.connect(path)) as connection, connection:
        (transcript,) = connection.execute(
            "SELECT data FROM items WHERE id = ?", (active.id,)
        ).fetchone()
    assert "call_active" not in transcript
    assert "resp_original" not in transcript
    assert "model_continuation" not in transcript
    reopened = SQLiteThreadStore[RequestContext](path)
    assert await reopened.load_item(thread.id, active.id, context) == active
    assert (
        await reopened.load_client_tool_invocation(thread.id, active.id, context)
        == invocation
    )
    assert (
        await reopened.load_client_tool_status(thread.id, active.id, context)
        == "pending"
    )
    with pytest.raises(NotFoundError):
        await reopened.load_client_tool_invocation(
            thread.id, active.id, ALTERNATIVE_CONTEXT
        )
    public = await reopened.load_thread_items(
        thread.id, None, 2, "asc", context, include_hidden=False
    )
    assert public.data == [active]
    assert not public.has_more

    completed = active.model_copy(update={"status": "completed", "output": "current"})
    with pytest.raises(ValueError, match="continuation turn ID"):
        await reopened.claim_pending_item(thread.id, active, completed, context)
    with pytest.raises(ValueError, match="only complete the item with output"):
        await reopened.claim_pending_item(
            thread.id,
            active,
            completed.model_copy(update={"arguments": {"value": "changed"}}),
            context,
            next_turn_id="next",
        )
    claimed = await reopened.claim_pending_item(
        thread.id, active, completed, context, next_turn_id="next"
    )
    expected = invocation.model_copy(update={"continuation_turn_id": "next"})
    assert claimed == expected
    assert (
        await initial.load_client_tool_invocation(thread.id, active.id, context)
        == expected
    )
    assert (
        await reopened.load_turn_state(thread.id, "parent", context)
        is TurnState.WAITING
    )
    assert (
        await reopened.load_turn_state(thread.id, "next", context) is TurnState.ACCEPTED
    )
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
    await first_store.begin_turn(thread.id, "parent", context, create_thread=thread)
    await first_store.start_turn(thread.id, "parent", context)
    barrier = Barrier(2, timeout=5)

    def register(store: SQLiteThreadStore[RequestContext], item_id: str) -> bool:
        item = ClientToolCallItem(
            id=item_id,
            thread_id=thread.id,
            created_at=thread.created_at,
            name="example",
            arguments={},
        )
        barrier.wait()
        try:
            asyncio.run(
                store.register_client_tool_call(
                    thread.id, item, tool_invocation(item, "parent"), context
                )
            )
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
    loser = ("first", "second")[results.index(False)]
    with pytest.raises(NotFoundError):
        await first_store.load_client_tool_invocation(thread.id, loser, context)

    await first_store.finish_turn(thread.id, "parent", TurnOutcome.FAILED, context)
    assert (
        await second_store.load_client_tool_status(thread.id, winner, context)
        == "cancelled"
    )
    assert not (
        await second_store.load_thread_items(
            thread.id, None, 10, "asc", context, include_hidden=False
        )
    ).data


@pytest.mark.asyncio
async def test_in_memory_save_user_message_cancels_unfinished_callback():
    store = InMemoryThreadStore()
    context: dict[str, object] = {}
    thread = make_thread()
    await store.begin_turn(thread.id, "parent", context, create_thread=thread)
    await store.start_turn(thread.id, "parent", context)
    tool = ClientToolCallItem(
        id="tool",
        thread_id=thread.id,
        created_at=thread.created_at,
        name="example",
        arguments={},
    )
    continuation = ClientToolResponsesContinuation(
        response_id="resp_private", items=[{"example": "original"}]
    )
    invocation = tool_invocation(tool, "parent").model_copy(
        update={"model_continuation": continuation}
    )
    await store.register_client_tool_call(thread.id, tool, invocation, context)
    continuation.items[0]["example"] = "changed after registration"
    saved = await store.load_client_tool_invocation(thread.id, tool.id, context)
    assert isinstance(saved.model_continuation, ClientToolResponsesContinuation)
    assert saved.model_continuation.items == [{"example": "original"}]
    saved.model_continuation.items[0]["example"] = "changed after reading"
    reloaded = await store.load_client_tool_invocation(thread.id, tool.id, context)
    assert isinstance(reloaded.model_continuation, ClientToolResponsesContinuation)
    assert reloaded.model_continuation.items == [{"example": "original"}]
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

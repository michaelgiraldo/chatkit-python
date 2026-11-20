import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

import pytest
from pydantic import AnyUrl

from chatkit.store import (
    AttachmentMetadataStore,
    ConflictError,
    NotFoundError,
    PendingTurnItem,
    ThreadStore,
)
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ClientToolCallItem,
    FileAttachment,
    ImageAttachment,
    InferenceOptions,
    LockedStatus,
    Page,
    StructuredInputItem,
    ThreadItem,
    ThreadMetadata,
    UserMessageItem,
    UserMessageTextContent,
    WidgetItem,
)
from chatkit.widgets import Card, Col, Text
from tests._types import RequestContext


def make_thread(
    thread_id: str = "test_thread", created_at: datetime | None = None
) -> ThreadMetadata:
    if created_at is None:
        created_at = datetime.now()
    return ThreadMetadata(
        id=thread_id,
        title="Test Thread",
        created_at=created_at,
        metadata={"test": "test"},
    )


def make_thread_items(thread_id: str) -> list[ThreadItem]:
    now = datetime.now()
    user_msg = UserMessageItem(
        id="msg_100000",
        content=[UserMessageTextContent(text="Hello!")],
        attachments=[
            FileAttachment(
                id="file_1",
                type="file",
                mime_type="text/plain",
                name="test.txt",
                metadata={"source": "test"},
            )
        ],
        inference_options=InferenceOptions(),
        thread_id=thread_id,
        created_at=now,
    )
    assistant_msg = AssistantMessageItem(
        id="msg_000001",
        content=[AssistantMessageContent(text="Hi there!")],
        thread_id=thread_id,
        created_at=now + timedelta(seconds=1),
    )
    widget = WidgetItem(
        id="widget_1",
        type="widget",
        thread_id=thread_id,
        created_at=now + timedelta(seconds=2),
        widget=Card(
            children=[
                Col(
                    padding={"x": 4, "y": 3},
                    border={"bottom": 1},
                    children=[
                        Text(
                            value="Title",
                            weight="medium",
                            size="sm",
                            color="secondary",
                        ),
                        Text(
                            value="test",
                        ),
                    ],
                ),
            ]
        ),
    )

    return [user_msg, assistant_msg, widget]


DEFAULT_CONTEXT = RequestContext(user_id="test_user")
ALTERNATIVE_CONTEXT = RequestContext(user_id="alternative_user")


async def assert_tied_timestamp_pagination[T: ThreadMetadata | ThreadItem](
    load_page: Callable[[str | None, int, str], Awaitable[Page[T]]],
    expected_ids: set[str],
) -> None:
    ascending = await load_page(None, len(expected_ids), "asc")
    descending = await load_page(None, len(expected_ids), "desc")
    assert {row.id for row in ascending.data} == expected_ids
    assert len(ascending.data) == len(expected_ids)
    assert descending.data == list(reversed(ascending.data))

    for order, unpaginated in (("asc", ascending), ("desc", descending)):
        cursor = None
        for position, row in enumerate(unpaginated.data):
            page = await load_page(cursor, 1, order)
            assert page.data == [row]
            assert page.has_more == (position < len(expected_ids) - 1)
            assert page.after == (row.id if page.has_more else None)
            cursor = page.after


class ThreadStoreContract(ABC):
    """Inherit these tests and supply a fresh thread store for each test."""

    store: ThreadStore[RequestContext]

    @abstractmethod
    def setup_method(self, method):
        pass

    @abstractmethod
    def teardown_method(self, method):
        pass

    @pytest.mark.asyncio
    async def test_thread_patch_changes_only_selected_fields_and_keys(self):
        thread = make_thread()
        thread.allowed_image_domains = ["example.com"]
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        await self.store.patch_thread(
            thread.id,
            {"title": "Renamed", "status": LockedStatus(), "metadata_set": {"keep": 1}},
            DEFAULT_CONTEXT,
        )
        patched = await self.store.patch_thread(
            thread.id,
            {
                "allowed_image_domains": None,
                "metadata_delete": ("test",),
                "metadata_set": {"response_id": "response_1", "nullable": None},
            },
            DEFAULT_CONTEXT,
        )
        assert patched.title == "Renamed"
        assert patched.status == LockedStatus()
        assert patched.allowed_image_domains is None
        assert patched.created_at == thread.created_at
        assert patched.metadata == {
            "keep": 1,
            "response_id": "response_1",
            "nullable": None,
        }
        assert await self.store.load_thread(thread.id, DEFAULT_CONTEXT) == patched
        cleared = await self.store.patch_thread(
            thread.id, {"title": None}, DEFAULT_CONTEXT
        )
        assert cleared.title is None
        with pytest.raises(NotFoundError):
            await self.store.patch_thread(
                thread.id, {"title": "Other"}, ALTERNATIVE_CONTEXT
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", [False, True])
    async def test_pending_item_can_only_be_claimed_once(self, tool: bool):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        pending: PendingTurnItem
        if tool:
            pending = ClientToolCallItem(
                id="pending",
                thread_id=thread.id,
                created_at=thread.created_at,
                call_id="call_example",
                name="example",
                arguments={},
            )
            first = pending.model_copy(update={"status": "completed", "output": 1})
            second = pending.model_copy(update={"status": "completed", "output": 2})
        else:
            pending = StructuredInputItem(
                id="pending",
                thread_id=thread.id,
                created_at=thread.created_at,
                inputs=[],
            )
            first = pending.model_copy(update={"status": "answered"})
            second = pending.model_copy(update={"status": "skipped"})
        await self.store.add_thread_item(thread.id, pending, DEFAULT_CONTEXT)
        with pytest.raises(NotFoundError):
            await self.store.claim_pending_item(
                thread.id, pending, first, ALTERNATIVE_CONTEXT
            )
        results = await asyncio.gather(
            self.store.claim_pending_item(thread.id, pending, first, DEFAULT_CONTEXT),
            self.store.claim_pending_item(thread.id, pending, second, DEFAULT_CONTEXT),
            return_exceptions=True,
        )
        assert results.count(None) == 1
        assert sum(isinstance(result, ConflictError) for result in results) == 1
        winner = (first, second)[results.index(None)]
        assert (
            await self.store.load_item(thread.id, pending.id, DEFAULT_CONTEXT) == winner
        )

    @pytest.mark.asyncio
    async def test_client_tool_claim_checks_latest_item_in_stable_order(self):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        pending = ClientToolCallItem(
            id="a",
            thread_id=thread.id,
            created_at=thread.created_at,
            call_id="call_example",
            name="example",
            arguments={},
        )
        await self.store.add_thread_item(thread.id, pending, DEFAULT_CONTEXT)
        await self.store.add_thread_item(
            thread.id,
            AssistantMessageItem(
                id="b", thread_id=thread.id, created_at=thread.created_at, content=[]
            ),
            DEFAULT_CONTEXT,
        )
        completed = pending.model_copy(update={"status": "completed", "output": "ok"})
        with pytest.raises(ConflictError):
            await self.store.claim_pending_item(
                thread.id, pending, completed, DEFAULT_CONTEXT
            )
        assert (
            await self.store.load_item(thread.id, pending.id, DEFAULT_CONTEXT)
            == pending
        )

    @pytest.mark.asyncio
    async def test_truncate_after_retries_from_user_message_in_stable_order(self):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        target = UserMessageItem(
            id="b",
            thread_id=thread.id,
            created_at=thread.created_at,
            content=[],
            inference_options=InferenceOptions(),
        )
        for item_id in ("d", "a", "c"):
            await self.store.add_thread_item(
                thread.id,
                AssistantMessageItem(
                    id=item_id,
                    thread_id=thread.id,
                    created_at=thread.created_at,
                    content=[],
                ),
                DEFAULT_CONTEXT,
            )
        await self.store.add_thread_item(thread.id, target, DEFAULT_CONTEXT)

        with pytest.raises(ConflictError):
            await self.store.truncate_after(thread.id, "a", DEFAULT_CONTEXT)
        with pytest.raises(NotFoundError):
            await self.store.truncate_after(thread.id, target.id, ALTERNATIVE_CONTEXT)
        assert await self.store.truncate_after(
            thread.id, target.id, DEFAULT_CONTEXT
        ) == (
            target,
            ["d", "c"],
        )
        surviving_items = await self.store.load_thread_items(
            thread.id,
            None,
            10,
            "asc",
            DEFAULT_CONTEXT,
        )
        assert [item.id for item in surviving_items.data] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_save_and_load_thread_metadata(self):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        loaded_meta = await self.store.load_thread(thread.id, DEFAULT_CONTEXT)
        assert loaded_meta.id == thread.id
        assert loaded_meta.title == thread.title
        assert loaded_meta.metadata == thread.metadata

    @pytest.mark.asyncio
    async def test_save_and_load_thread_metadata_null_title(self):
        thread = ThreadMetadata(
            id="test_thread",
            title=None,
            created_at=datetime.now(),
            metadata={"test": "test"},
        )
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        loaded_meta = await self.store.load_thread(thread.id, DEFAULT_CONTEXT)
        assert loaded_meta.id == thread.id
        assert loaded_meta.title is None
        assert loaded_meta.metadata == thread.metadata

    @pytest.mark.asyncio
    async def test_update_thread_metadata(self):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        thread.title = "Updated Title"
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        loaded_meta = await self.store.load_thread(thread.id, DEFAULT_CONTEXT)
        assert loaded_meta.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_save_and_load_thread_items(self):
        thread = make_thread()
        items = make_thread_items(thread.id)
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        for item in items:
            await self.store.add_thread_item(thread.id, item, DEFAULT_CONTEXT)
        loaded_items = (
            await self.store.load_thread_items(
                thread.id, None, 10, "asc", DEFAULT_CONTEXT
            )
        ).data
        assert loaded_items == items

    @pytest.mark.asyncio
    async def test_overwrite_thread_metadata(self):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        thread.title = "Updated Title"
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        loaded_meta = await self.store.load_thread(thread.id, DEFAULT_CONTEXT)
        assert loaded_meta.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_load_threads(self):
        now = datetime.now()
        thread1 = make_thread(thread_id="thread1", created_at=now)
        thread2 = make_thread(
            thread_id="thread2", created_at=now + timedelta(seconds=1)
        )
        thread3 = make_thread(
            thread_id="thread3", created_at=now + timedelta(seconds=2)
        )
        await self.store.save_thread(thread1, DEFAULT_CONTEXT)
        await self.store.save_thread(thread2, DEFAULT_CONTEXT)
        await self.store.save_thread(thread3, DEFAULT_CONTEXT)

        page1 = await self.store.load_threads(
            limit=2, after=None, order="asc", context=DEFAULT_CONTEXT
        )
        assert [t.id for t in page1.data] == ["thread1", "thread2"]
        assert page1.has_more is True
        assert page1.after == "thread2"

        page2 = await self.store.load_threads(
            limit=2, after=page1.data[-1].id, order="asc", context=DEFAULT_CONTEXT
        )
        assert [t.id for t in page2.data] == ["thread3"]
        assert page2.has_more is False
        assert not page2.after

    @pytest.mark.asyncio
    async def test_load_threads_paginates_tied_timestamps(self):
        created_at = datetime.now()
        thread_ids = {"thread-a", "thread-b", "thread-c"}
        for thread_id in sorted(thread_ids, reverse=True):
            await self.store.save_thread(
                make_thread(thread_id, created_at), DEFAULT_CONTEXT
            )

        async def load_page(after: str | None, limit: int, order: str):
            return await self.store.load_threads(limit, after, order, DEFAULT_CONTEXT)

        await assert_tied_timestamp_pagination(load_page, thread_ids)

    @pytest.mark.asyncio
    async def test_thread_items_ordering(self):
        thread = make_thread()
        now = datetime.now()
        items = [
            UserMessageItem(
                id="msg1",
                content=[UserMessageTextContent(text="A")],
                attachments=[],
                inference_options=InferenceOptions(),
                thread_id=thread.id,
                created_at=now,
            ),
            UserMessageItem(
                id="msg2",
                content=[UserMessageTextContent(text="B")],
                attachments=[],
                inference_options=InferenceOptions(),
                thread_id=thread.id,
                created_at=now + timedelta(seconds=1),
            ),
            UserMessageItem(
                id="msg3",
                content=[UserMessageTextContent(text="C")],
                attachments=[],
                inference_options=InferenceOptions(),
                thread_id=thread.id,
                created_at=now + timedelta(seconds=2),
            ),
        ]
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        for item in items:
            await self.store.add_thread_item(thread.id, item, DEFAULT_CONTEXT)

        asc = await self.store.load_thread_items(
            thread.id, after=None, limit=3, order="asc", context=DEFAULT_CONTEXT
        )
        desc = await self.store.load_thread_items(
            thread.id, after=None, limit=3, order="desc", context=DEFAULT_CONTEXT
        )
        assert [i.id for i in asc.data] == ["msg1", "msg2", "msg3"]
        assert [i.id for i in desc.data] == ["msg3", "msg2", "msg1"]

    @pytest.mark.asyncio
    async def test_thread_items_offset(self):
        thread = make_thread()
        now = datetime.now()
        items = [
            UserMessageItem(
                id=f"msg{i}",
                content=[UserMessageTextContent(text=str(i))],
                attachments=[],
                inference_options=InferenceOptions(),
                thread_id=thread.id,
                created_at=now + timedelta(seconds=i),
            )
            for i in range(5)
        ]
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        for item in items:
            await self.store.add_thread_item(thread.id, item, DEFAULT_CONTEXT)

        after = await self.store.load_thread_items(
            thread.id, after="msg1", limit=3, order="asc", context=DEFAULT_CONTEXT
        )
        assert [i.id for i in after.data] == ["msg2", "msg3", "msg4"]
        assert after.has_more is False
        assert not after.after

        after_limit = await self.store.load_thread_items(
            thread.id, after=None, limit=2, order="asc", context=DEFAULT_CONTEXT
        )
        assert [i.id for i in after_limit.data] == ["msg0", "msg1"]
        assert after_limit.has_more is True
        assert after_limit.after == "msg1"

    @pytest.mark.asyncio
    async def test_load_thread_items_paginates_tied_timestamps(self):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        item_ids = {"msg-a", "msg-b", "msg-c"}
        for item_id in sorted(item_ids, reverse=True):
            await self.store.add_thread_item(
                thread.id,
                AssistantMessageItem(
                    id=item_id,
                    thread_id=thread.id,
                    created_at=thread.created_at,
                    content=[AssistantMessageContent(text=item_id)],
                ),
                DEFAULT_CONTEXT,
            )

        async def load_page(after: str | None, limit: int, order: str):
            return await self.store.load_thread_items(
                thread.id, after, limit, order, DEFAULT_CONTEXT
            )

        await assert_tied_timestamp_pagination(load_page, item_ids)

    @pytest.mark.asyncio
    async def test_save_and_load_item(self):
        thread = make_thread()
        now = datetime.now()
        assistant_msg = AssistantMessageItem(
            id="msg_000001",
            content=[],
            thread_id=thread.id,
            created_at=now,
        )
        widget = WidgetItem(
            id="widget_1",
            type="widget",
            thread_id=thread.id,
            created_at=now + timedelta(seconds=1),
            widget=Card(children=[Text(value="Test")]),
        )
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        await self.store.save_item(thread.id, assistant_msg, DEFAULT_CONTEXT)
        assert (
            await self.store.load_item(thread.id, assistant_msg.id, DEFAULT_CONTEXT)
            == assistant_msg
        )
        await self.store.add_thread_item(thread.id, widget, DEFAULT_CONTEXT)

        assistant_msg.content = [
            AssistantMessageContent(text="This is an assistant message.")
        ]
        await self.store.save_item(thread.id, assistant_msg, DEFAULT_CONTEXT)
        loaded_assistant = await self.store.load_item(
            thread.id, assistant_msg.id, DEFAULT_CONTEXT
        )
        assert isinstance(loaded_assistant, AssistantMessageItem)
        assert loaded_assistant.id == assistant_msg.id
        assert loaded_assistant.content[0] == AssistantMessageContent(
            text="This is an assistant message."
        )

        await self.store.save_item(thread.id, widget, DEFAULT_CONTEXT)
        loaded_widget = await self.store.load_item(
            thread.id, widget.id, DEFAULT_CONTEXT
        )
        assert isinstance(loaded_widget, WidgetItem)
        assert loaded_widget.id == widget.id

        page = await self.store.load_thread_items(
            thread.id, None, 10, "asc", DEFAULT_CONTEXT
        )
        assert page.data == [assistant_msg, widget]

    @pytest.mark.asyncio
    async def test_load_nonexistent_item(self):
        thread = make_thread()
        await self.store.save_thread(thread, DEFAULT_CONTEXT)
        with pytest.raises(NotFoundError):
            await self.store.load_item(thread.id, "does_not_exist", DEFAULT_CONTEXT)

    @pytest.mark.asyncio
    async def test_thread_isolation_by_user(self):
        # Create a thread for DEFAULT_CONTEXT
        thread = make_thread(thread_id="user1_thread")
        await self.store.save_thread(thread, DEFAULT_CONTEXT)

        # Should be accessible by the same user
        loaded_default = await self.store.load_thread(thread.id, DEFAULT_CONTEXT)
        assert loaded_default.title == thread.title

        # Should not be accessible by another user
        with pytest.raises(NotFoundError):
            await self.store.load_thread(thread.id, ALTERNATIVE_CONTEXT)

    @pytest.mark.asyncio
    async def test_thread_items_isolation_by_user(self):
        thread = make_thread(thread_id="shared_thread")
        await self.store.save_thread(thread, DEFAULT_CONTEXT)

        # Add items for DEFAULT_CONTEXT
        items_default = make_thread_items(thread.id)
        for item in items_default:
            await self.store.add_thread_item(thread.id, item, DEFAULT_CONTEXT)

        # Should be accessible by the same user
        loaded_items_default = (
            await self.store.load_thread_items(
                thread.id, None, 10, "asc", DEFAULT_CONTEXT
            )
        ).data
        assert loaded_items_default == items_default

        # Should not be accessible by another user (should return empty list)
        loaded_items_alternative = (
            await self.store.load_thread_items(
                thread.id, None, 10, "asc", ALTERNATIVE_CONTEXT
            )
        ).data
        assert loaded_items_alternative == []

        # Try to load a specific item with another user's context.
        with pytest.raises(NotFoundError):
            await self.store.load_item(
                thread.id, items_default[0].id, ALTERNATIVE_CONTEXT
            )


class AttachmentMetadataStoreContract:
    """Inherit these tests when the backend implements attachment metadata."""

    attachment_metadata_store: AttachmentMetadataStore[RequestContext]

    @pytest.mark.asyncio
    async def test_save_and_load_file(self):
        file = FileAttachment(
            id="file_1",
            type="file",
            mime_type="text/plain",
            name="test.txt",
        )
        await self.attachment_metadata_store.save_attachment(file, DEFAULT_CONTEXT)
        loaded = await self.attachment_metadata_store.load_attachment(
            file.id, DEFAULT_CONTEXT
        )
        assert loaded == file

    @pytest.mark.asyncio
    async def test_save_and_load_image(self):
        image = ImageAttachment(
            id="image_1",
            type="image",
            mime_type="image/png",
            name="test.png",
            preview_url=AnyUrl("https://example.com/test.png"),
            metadata={"source": "test"},
        )
        await self.attachment_metadata_store.save_attachment(image, DEFAULT_CONTEXT)
        loaded = await self.attachment_metadata_store.load_attachment(
            image.id, DEFAULT_CONTEXT
        )
        assert loaded == image

    @pytest.mark.asyncio
    async def test_attachment_metadata_is_scoped_and_upserted(self):
        store = self.attachment_metadata_store
        file = FileAttachment(
            id="file_1",
            type="file",
            mime_type="text/plain",
            name="test.txt",
        )
        await store.save_attachment(file, DEFAULT_CONTEXT)
        file.name = "updated.txt"
        await store.save_attachment(file, DEFAULT_CONTEXT)
        await store.delete_attachment(file.id, ALTERNATIVE_CONTEXT)
        assert await store.load_attachment(file.id, DEFAULT_CONTEXT) == file
        with pytest.raises(NotFoundError):
            await store.load_attachment(file.id, ALTERNATIVE_CONTEXT)

        await store.delete_attachment(file.id, DEFAULT_CONTEXT)
        with pytest.raises(NotFoundError):
            await store.load_attachment(file.id, DEFAULT_CONTEXT)

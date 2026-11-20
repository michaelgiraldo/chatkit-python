from collections.abc import Iterable
from datetime import UTC, datetime

from chatkit.models import (
    ClientToolCallItem,
    Page,
    ThreadItem,
    ThreadMetadata,
    UserMessageItem,
)
from chatkit.store import (
    HIDDEN_ITEM_TYPES,
    ConflictError,
    NotFoundError,
    PendingTurnItem,
    ThreadPatch,
    ThreadStore,
    apply_thread_patch,
    validate_pending_claim,
)

type Context = dict[str, object]


class InMemoryThreadStore(ThreadStore[Context]):
    """Store threads and items on one local development event loop.

    Mutations run without awaiting so that each one is atomic on that loop.
    """

    def __init__(self):
        self.threads: dict[str, ThreadMetadata] = {}
        self.items: dict[str, dict[str, ThreadItem]] = {}

    async def load_thread(self, thread_id: str, context: Context) -> ThreadMetadata:
        if thread_id not in self.threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        return self.threads[thread_id].model_copy(deep=True)

    async def save_thread(self, thread: ThreadMetadata, context: Context) -> None:
        self.threads[thread.id] = thread.model_copy(deep=True)

    async def patch_thread(
        self, thread_id: str, patch: ThreadPatch, context: Context
    ) -> ThreadMetadata:
        current = self.threads.get(thread_id)
        if current is None:
            raise NotFoundError(f"Thread {thread_id} not found")
        updated = apply_thread_patch(current, patch)
        self.threads[thread_id] = updated.model_copy(deep=True)
        return updated.model_copy(deep=True)

    async def load_threads(
        self, limit: int, after: str | None, order: str, context: Context
    ) -> Page[ThreadMetadata]:
        return self._page(self.threads.values(), after, limit, order)

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: Context,
        *,
        include_hidden: bool = True,
    ) -> Page[ThreadItem]:
        items = self.items.get(thread_id, {}).values()
        rows = (
            items
            if include_hidden
            else (item for item in items if item.type not in HIDDEN_ITEM_TYPES)
        )
        return self._page(rows, after, limit, order)

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: Context
    ) -> None:
        items = self.items.setdefault(thread_id, {})
        if item.id in items:
            raise ValueError(f"Item {item.id} already exists in thread {thread_id}")
        items[item.id] = item.model_copy(deep=True)

    async def save_item(
        self, thread_id: str, item: ThreadItem, context: Context
    ) -> None:
        self.items.setdefault(thread_id, {})[item.id] = item.model_copy(deep=True)

    async def claim_pending_item(
        self,
        thread_id: str,
        expected: PendingTurnItem,
        replacement: PendingTurnItem,
        context: Context,
    ) -> None:
        if thread_id not in self.threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        items = self.items.get(thread_id, {})
        current = items.get(expected.id)
        if current is None:
            raise NotFoundError(f"Item {expected.id} not found in thread {thread_id}")
        latest_item_id = None
        if isinstance(expected, ClientToolCallItem):
            latest_item_id = max(items.values(), key=self._sort_key).id
        validate_pending_claim(
            current, expected, replacement, latest_item_id=latest_item_id
        )
        items[expected.id] = replacement.model_copy(deep=True)

    async def truncate_after(
        self, thread_id: str, item_id: str, context: Context
    ) -> tuple[UserMessageItem, list[str]]:
        if thread_id not in self.threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        items = self.items.get(thread_id, {})
        item = items.get(item_id)
        if item is None:
            raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
        if not isinstance(item, UserMessageItem):
            raise ConflictError("A retry must start from a user message")
        target = self._sort_key(item)
        later = sorted(
            (row for row in items.values() if self._sort_key(row) > target),
            key=self._sort_key,
            reverse=True,
        )
        for row in later:
            del items[row.id]
        return item.model_copy(deep=True), [row.id for row in later]

    async def load_item(
        self, thread_id: str, item_id: str, context: Context
    ) -> ThreadItem:
        item = self.items.get(thread_id, {}).get(item_id)
        if item is None:
            raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
        return item.model_copy(deep=True)

    async def delete_thread(self, thread_id: str, context: Context) -> None:
        self.threads.pop(thread_id, None)
        self.items.pop(thread_id, None)

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: Context
    ) -> None:
        self.items.get(thread_id, {}).pop(item_id, None)

    @staticmethod
    def _sort_key(row: ThreadMetadata | ThreadItem) -> tuple[datetime, str]:
        return row.created_at.astimezone(UTC), row.id

    def _page[T: ThreadMetadata | ThreadItem](
        self, rows: Iterable[T], after: str | None, limit: int, order: str
    ) -> Page[T]:
        ordered = sorted(rows, key=self._sort_key, reverse=order == "desc")
        start = 0
        if after is not None:
            for index, row in enumerate(ordered):
                if row.id == after:
                    start = index + 1
                    break
            else:
                raise NotFoundError(f"Cursor {after} not found")

        data = ordered[start : start + limit]
        has_more = bool(data) and start + len(data) < len(ordered)
        return Page(
            data=[row.model_copy(deep=True) for row in data],
            has_more=has_more,
            after=data[-1].id if has_more else None,
        )

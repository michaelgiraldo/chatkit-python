from collections.abc import Iterable

from chatkit.store import NotFoundError, ThreadStore
from chatkit.types import Page, ThreadItem, ThreadMetadata

type Context = dict[str, object]


class InMemoryThreadStore(ThreadStore[Context]):
    """Store threads and items for a single local development process."""

    def __init__(self):
        self.threads: dict[str, ThreadMetadata] = {}
        self.items: dict[str, dict[str, ThreadItem]] = {}

    async def load_thread(self, thread_id: str, context: Context) -> ThreadMetadata:
        if thread_id not in self.threads:
            raise NotFoundError(f"Thread {thread_id} not found")
        return self.threads[thread_id].model_copy(deep=True)

    async def save_thread(self, thread: ThreadMetadata, context: Context) -> None:
        self.threads[thread.id] = thread.model_copy(deep=True)

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
    ) -> Page[ThreadItem]:
        return self._page(self.items.get(thread_id, {}).values(), after, limit, order)

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

    def _page[T: ThreadMetadata | ThreadItem](
        self, rows: Iterable[T], after: str | None, limit: int, order: str
    ) -> Page[T]:
        ordered = sorted(
            rows, key=lambda row: (row.created_at, row.id), reverse=order == "desc"
        )
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

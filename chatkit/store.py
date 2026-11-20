import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal

from .types import (
    Attachment,
    AttachmentCreateParams,
    Page,
    ThreadItem,
    ThreadMetadata,
)

StoreItemType = Literal[
    "thread",
    "message",
    "tool_call",
    "task",
    "workflow",
    "attachment",
    "sdk_hidden_context",
]


_ID_PREFIXES: dict[StoreItemType, str] = {
    "thread": "thr",
    "message": "msg",
    "tool_call": "tc",
    "workflow": "wf",
    "task": "tsk",
    "attachment": "atc",
    "sdk_hidden_context": "shcx",
}


def default_generate_id(item_type: StoreItemType) -> str:
    prefix = _ID_PREFIXES[item_type]
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class NotFoundError(Exception):
    pass


class AttachmentStore[TContext = Any](ABC):
    """Handle attachment uploads and delete the associated file data."""

    @abstractmethod
    async def delete_attachment(self, attachment_id: str, context: TContext) -> None:
        """Delete an attachment by id."""
        pass

    async def create_attachment(
        self, input: AttachmentCreateParams, context: TContext
    ) -> Attachment:
        """Create an attachment record from upload metadata."""
        raise NotImplementedError(
            f"{type(self).__name__} must override create_attachment() to support two-phase file upload"
        )

    def generate_attachment_id(self, mime_type: str, context: TContext) -> str:
        """Return a new identifier for a file. Override this method to customize file ID generation."""

        return default_generate_id("attachment")


class ThreadStore[TContext = Any](ABC):
    """Persist threads and their items, with access scoped by the request context."""

    def generate_thread_id(self, context: TContext) -> str:
        """Return a new identifier for a thread. Override this method to customize thread ID generation."""

        return default_generate_id("thread")

    def generate_item_id(
        self, item_type: StoreItemType, thread: ThreadMetadata, context: TContext
    ) -> str:
        """Return a new identifier for a thread item. Override this method to customize item ID generation."""

        return default_generate_id(item_type)

    @abstractmethod
    async def load_thread(self, thread_id: str, context: TContext) -> ThreadMetadata:
        """Load thread metadata or raise `NotFoundError` if it is unavailable."""
        pass

    @abstractmethod
    async def save_thread(self, thread: ThreadMetadata, context: TContext) -> None:
        """Insert or update thread metadata by id."""
        pass

    @abstractmethod
    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: TContext,
    ) -> Page[ThreadItem]:
        """Load up to `limit` items in creation order, starting after `after`.

        Use a stable order for items with the same timestamp. Accept `"asc"`
        for oldest first and `"desc"` for newest first. If more items exist,
        set `has_more=True` and return a cursor in `Page.after` for the next page.
        """
        pass

    @abstractmethod
    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: TContext,
    ) -> Page[ThreadMetadata]:
        """Load up to `limit` threads in creation order, starting after `after`.

        Use a stable order for threads with the same timestamp. Accept `"asc"`
        for oldest first and `"desc"` for newest first. If more threads exist,
        set `has_more=True` and return a cursor in `Page.after` for the next page.
        """
        pass

    @abstractmethod
    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: TContext
    ) -> None:
        """Persist a newly created thread item."""
        pass

    @abstractmethod
    async def save_item(
        self, thread_id: str, item: ThreadItem, context: TContext
    ) -> None:
        """Upsert a thread item by id."""
        pass

    @abstractmethod
    async def load_item(
        self, thread_id: str, item_id: str, context: TContext
    ) -> ThreadItem:
        """Load an item in the given thread or raise `NotFoundError`."""
        pass

    @abstractmethod
    async def delete_thread(self, thread_id: str, context: TContext) -> None:
        """Delete a thread and its items."""
        pass

    @abstractmethod
    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: TContext
    ) -> None:
        """Delete a thread item by id."""
        pass


class AttachmentMetadataStore[TContext = Any](ABC):
    """Persist attachment records, with access scoped by the request context."""

    @abstractmethod
    async def save_attachment(self, attachment: Attachment, context: TContext) -> None:
        """Upsert attachment metadata."""
        pass

    @abstractmethod
    async def load_attachment(
        self, attachment_id: str, context: TContext
    ) -> Attachment:
        """Load attachment metadata or raise `NotFoundError`."""
        pass

    @abstractmethod
    async def delete_attachment(self, attachment_id: str, context: TContext) -> None:
        """Delete attachment metadata by id."""
        pass

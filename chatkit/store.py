import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict

from .models import (
    Attachment,
    ClientToolCallItem,
    Page,
    StructuredInputItem,
    ThreadItem,
    ThreadMetadata,
    ThreadStatus,
    UserMessageItem,
)
from .requests import AttachmentCreateParams

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
    return f"{prefix}_{uuid.uuid4().hex}"


HIDDEN_ITEM_TYPES = ("hidden_context_item", "sdk_hidden_context")


class NotFoundError(Exception):
    pass


class ConflictError(Exception):
    """The stored state no longer permits the requested operation."""


type PendingTurnItem = ClientToolCallItem | StructuredInputItem


class ThreadPatch(TypedDict, total=False):
    """Explicit changes to thread display fields or individual metadata keys."""

    title: str | None
    status: ThreadStatus
    allowed_image_domains: list[str] | None
    metadata_set: dict[str, Any]
    metadata_delete: tuple[str, ...]


def apply_thread_patch(thread: ThreadMetadata, patch: ThreadPatch) -> ThreadMetadata:
    """Build an updated snapshot from a store's current thread and a patch."""
    fields = {
        key: value
        for key, value in patch.items()
        if key in ("title", "status", "allowed_image_domains")
    }
    metadata = {
        key: value
        for key, value in thread.metadata.items()
        if key not in patch.get("metadata_delete", ())
    }
    metadata.update(patch.get("metadata_set", {}))
    return thread.model_copy(update={**fields, "metadata": metadata})


def validate_pending_claim(
    current: ThreadItem,
    expected: PendingTurnItem,
    replacement: PendingTurnItem,
    *,
    latest_item_id: str | None = None,
) -> None:
    """Validate a claim against the current state inside a store transaction."""
    if (
        replacement.id != expected.id
        or replacement.thread_id != expected.thread_id
        or replacement.created_at != expected.created_at
        or replacement.type != expected.type
        or replacement.status == "pending"
    ):
        raise ValueError("A pending claim must complete the same thread item")
    if current != expected or expected.status != "pending":
        raise ConflictError(
            "The thread item is no longer pending in the expected state"
        )
    if isinstance(expected, ClientToolCallItem) and expected.id != latest_item_id:
        raise ConflictError("The client tool call is no longer the latest thread item")


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
        """Save an initial thread snapshot. Use `patch_thread` for live edits."""
        pass

    @abstractmethod
    async def patch_thread(
        self, thread_id: str, patch: ThreadPatch, context: TContext
    ) -> ThreadMetadata:
        """Atomically apply a patch to the latest stored thread and return the result.

        Omitted display fields are left alone. Delete metadata keys before applying
        `metadata_set`. Raise `NotFoundError` for an unavailable thread.
        """
        pass

    @abstractmethod
    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: TContext,
        *,
        include_hidden: bool = True,
    ) -> Page[ThreadItem]:
        """Load up to `limit` items in creation order, starting after `after`.

        Use a stable order for items with the same timestamp. Accept `"asc"`
        for oldest first and `"desc"` for newest first. If more items exist,
        set `has_more=True` and return a cursor in `Page.after` for the next page.
        When `include_hidden=False`, exclude `HIDDEN_ITEM_TYPES` before applying
        the cursor and limit. Derive pagination metadata from the visible rows.
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
    async def claim_pending_item(
        self,
        thread_id: str,
        expected: PendingTurnItem,
        replacement: PendingTurnItem,
        context: TContext,
    ) -> None:
        """Atomically replace a pending item if its stored value matches `expected`.

        The replacement must complete the same thread item. A client tool call must
        still be the last item in the thread's stable order. Raise `ConflictError`
        on stale or completed state and `NotFoundError` for an unavailable item or
        thread. Commit the claim before starting inference.
        """
        pass

    @abstractmethod
    async def truncate_after(
        self, thread_id: str, item_id: str, context: TContext
    ) -> UserMessageItem:
        """Atomically truncate the active history after a stored user message.

        Use the stable order of `load_thread_items` and return the retained
        user message. Later items must disappear from pagination and direct
        reads before returning. A store can commit a visibility boundary and
        reclaim the inactive records separately.

        Raise `NotFoundError` for an unavailable target and `ConflictError`
        when the target is not a user message.
        """
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

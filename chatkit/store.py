import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict

from .errors import UnsupportedOperationError
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
from .turns import ClientToolInvocation, TurnOutcome, TurnState

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


class TurnSupersededError(ConflictError):
    """Another turn owns the thread, or the thread no longer exists."""


type PendingTurnItem = ClientToolCallItem | StructuredInputItem
type ClientToolCallbackStatus = Literal["pending", "completed", "cancelled"]


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
    if isinstance(expected, ClientToolCallItem) and (
        not isinstance(replacement, ClientToolCallItem)
        or replacement.status != "completed"
        or replacement.name != expected.name
        or replacement.arguments != expected.arguments
    ):
        raise ValueError("A client tool claim may only complete the item with output")


def validate_client_tool_registration(
    thread_id: str, item: ClientToolCallItem, invocation: ClientToolInvocation
) -> None:
    """Validate correspondence before saving a public item and private invocation."""
    if (
        item.thread_id != thread_id
        or invocation.thread_id != thread_id
        or invocation.item_id != item.id
    ):
        raise ValueError(
            "The client tool item and invocation must identify the same thread item"
        )
    if (
        item.status != "pending"
        or item.output is not None
        or invocation.continuation_turn_id is not None
    ):
        raise ValueError("A new client tool call must be pending")


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
    async def begin_turn(
        self,
        thread_id: str,
        turn_id: str,
        context: TContext,
        *,
        message: UserMessageItem | None = None,
        create_thread: ThreadMetadata | None = None,
    ) -> None:
        """Atomically accept a new current turn and cancel pending callbacks.

        Save the optional new thread and user message in the same transaction.
        Preserve the previous turn's disposition and reject reused turn IDs.
        """
        pass

    @abstractmethod
    async def start_turn(self, thread_id: str, turn_id: str, context: TContext) -> None:
        """Atomically move the current accepted turn to running, once."""
        pass

    @abstractmethod
    async def require_current_turn(
        self, thread_id: str, turn_id: str, context: TContext
    ) -> None:
        """Raise `ConflictError` when this turn no longer owns the thread."""
        pass

    @abstractmethod
    async def load_turn_state(
        self, thread_id: str, turn_id: str, context: TContext
    ) -> TurnState:
        """Read a scoped turn's state, including after ownership changes."""
        pass

    @abstractmethod
    async def finish_turn(
        self, thread_id: str, turn_id: str, outcome: TurnOutcome, context: TContext
    ) -> TurnState:
        """Finish the current running turn and return its recorded disposition.

        Cancellation before start records abandonment. Previously finalized and
        superseded turns keep their state even when their producers finish later.
        A current turn that stops without waiting cancels its pending callback.
        """
        pass

    @abstractmethod
    async def patch_thread(
        self,
        thread_id: str,
        patch: ThreadPatch,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> ThreadMetadata:
        """Atomically apply a patch to the latest stored thread and return the result.

        Omitted display fields are left alone. Delete metadata keys before applying
        `metadata_set`. Raise `NotFoundError` for an unavailable thread.
        When `turn_id` is given, verify ownership within the write transaction.
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
        When `include_hidden=False`, exclude `HIDDEN_ITEM_TYPES` and pending
        client tool items whose callback is no longer pending or was never
        registered before applying the cursor and limit. Derive pagination
        metadata from the visible rows.
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
        self,
        thread_id: str,
        item: ThreadItem,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Persist a newly created thread item.

        Use `register_client_tool_call` to persist client tool items. Reject
        reused callback IDs, including IDs whose items were removed, with
        `ConflictError`. A new user message atomically cancels a pending callback.
        Verify `turn_id` ownership within the insertion transaction.
        """
        pass

    @abstractmethod
    async def save_item(
        self,
        thread_id: str,
        item: ThreadItem,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Upsert a thread item by id.

        For client tools, use `register_client_tool_call` for registration and
        `claim_pending_item` for completion. Reject direct writes to those
        items and their reserved IDs. A first insertion of a user message
        has the same cancellation semantics as `add_thread_item`.
        When `turn_id` is given, verify ownership within the write transaction.
        """
        pass

    async def register_client_tool_call(
        self,
        thread_id: str,
        item: ClientToolCallItem,
        invocation: ClientToolInvocation,
        context: TContext,
    ) -> None:
        """Atomically save a browser item and its matching private invocation.

        Require the invocation's turn to be current and running. The thread can
        have one pending callback. Reserved callback IDs remain reserved after
        completion, cancellation, and display item deletion.
        """
        raise UnsupportedOperationError(
            "Client tool callbacks require a compatible thread store"
        )

    async def load_client_tool_invocation(
        self, thread_id: str, item_id: str, context: TContext
    ) -> ClientToolInvocation:
        """Read a scoped private invocation and its continuation turn, if claimed.

        Raise `NotFoundError` for an unavailable or unregistered callback.
        """
        raise UnsupportedOperationError(
            "Client tool callbacks require a compatible thread store"
        )

    async def load_client_tool_status(
        self, thread_id: str, item_id: str, context: TContext
    ) -> ClientToolCallbackStatus:
        """Read a scoped callback's lifecycle, including after item deletion.

        Raise `NotFoundError` for an unavailable or unregistered callback.
        """
        raise UnsupportedOperationError(
            "Client tool callbacks require a compatible thread store"
        )

    async def cancel_client_tool_call(
        self,
        thread_id: str,
        item_id: str,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Atomically cancel one pending callback. Closed or unknown IDs have no effect.

        When `turn_id` is given, verify ownership within the write transaction.
        """
        raise UnsupportedOperationError(
            "Client tool callbacks require a compatible thread store"
        )

    @abstractmethod
    async def claim_pending_item(
        self,
        thread_id: str,
        expected: PendingTurnItem,
        replacement: PendingTurnItem,
        context: TContext,
        *,
        next_turn_id: str | None = None,
    ) -> ClientToolInvocation | None:
        """Atomically replace a pending item if its stored value matches `expected`.

        The replacement must complete the same thread item. A client tool claim
        requires `next_turn_id` and a registered pending callback with a current
        owner. Save its result and the accepted continuation turn in the same
        transaction and return the private invocation linked to that turn.
        A structured input claim returns `None`. Record which turn issued a
        form. If a different turn is ACCEPTED or RUNNING, reject the claim
        with `ConflictError` in the same transaction, leaving the form pending.
        A form with no recorded owner can be claimed when the thread is idle.

        Raise `ConflictError` on stale or closed state and `NotFoundError` for
        an unavailable item or thread. Commit the claim before starting inference.
        """
        pass

    @abstractmethod
    async def truncate_after(
        self,
        thread_id: str,
        item_id: str,
        context: TContext,
        *,
        next_turn_id: str | None = None,
    ) -> UserMessageItem:
        """Atomically truncate the active history after a stored user message.

        Use the stable order of `load_thread_items` and return the retained
        user message. Later items must disappear from pagination and direct
        reads before returning. Atomically cancel any pending client tool
        callback and retain its status. A store can commit a visibility
        boundary and reclaim the inactive records separately.
        When `next_turn_id` is provided, accept that turn in the same transaction.

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
        """Delete a thread and its associated item, callback, and turn records."""
        pass

    @abstractmethod
    async def delete_thread_item(
        self,
        thread_id: str,
        item_id: str,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Delete an item and atomically cancel its pending callback.

        When `turn_id` is given, verify ownership within the write transaction.
        """
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

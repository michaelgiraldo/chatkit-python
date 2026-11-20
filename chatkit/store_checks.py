"""Checks for application implementations of :class:`ThreadStore`."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from .models import (
    ClientToolCallItem,
    InferenceOptions,
    Page,
    SDKHiddenContextItem,
    StructuredInputItem,
    ThreadItem,
    ThreadMetadata,
    UserMessageItem,
    UserMessageTextContent,
)
from .store import ConflictError, NotFoundError, PendingTurnItem, ThreadStore


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _check_item_pagination(
    load_page: Callable[[str | None, int, str], Awaitable[Page[ThreadItem]]],
    item_ids: set[str],
) -> list[str]:
    ascending = await load_page(None, len(item_ids), "asc")
    expected = [item.id for item in ascending.data]
    _require(
        set(expected) == item_ids and len(expected) == len(item_ids),
        "Pagination must return each item with a tied timestamp once",
    )
    for order, ordered_ids in (("asc", expected), ("desc", expected[::-1])):
        cursor = None
        for index, item_id in enumerate(ordered_ids):
            page = await load_page(cursor, 1, order)
            more_items = index < len(ordered_ids) - 1
            _require(
                [item.id for item in page.data] == [item_id]
                and page.has_more == more_items
                and (page.after is not None) == more_items,
                f"Broken {order} item cursor at page {index + 1}",
            )
            cursor = page.after
    return expected


async def check_thread_store[TContext](
    store: ThreadStore[TContext],
    context: TContext,
    *,
    other_context: TContext,
    peer: ThreadStore[TContext] | None = None,
) -> None:
    """Check scoped reads, pagination, patches, claims, and retry truncation.

    Run against a disposable database with contexts for two distinct users.
    Supply an independent store instance sharing that database to check
    concurrent operations across connections. The check writes and deletes
    a synthetic thread. A failed condition raises an informative AssertionError.
    """
    peer = peer if peer is not None else store
    created_at = datetime.now(UTC)
    thread = ThreadMetadata(
        id=store.generate_thread_id(context),
        created_at=created_at,
        metadata={"kept": True, "remove": True},
    )
    await store.save_thread(thread, context)
    try:
        try:
            await peer.load_thread(thread.id, other_context)
        except NotFoundError:
            pass
        else:
            raise AssertionError("Another user's context can load this thread")

        async with asyncio.TaskGroup() as patches:
            patches.create_task(
                store.patch_thread(thread.id, {"title": "Renamed"}, context)
            )
            patches.create_task(
                peer.patch_thread(
                    thread.id,
                    {"metadata_set": {"saved": True}, "metadata_delete": ("remove",)},
                    context,
                )
            )
        patched = await store.load_thread(thread.id, context)
        _require(
            patched.title == "Renamed"
            and patched.metadata == {"kept": True, "saved": True},
            "Concurrent field patches must preserve unrelated state",
        )

        item_ids = set()
        for text in ("First message", "Second message", "Third message"):
            message = UserMessageItem(
                id=store.generate_item_id("message", thread, context),
                thread_id=thread.id,
                created_at=created_at,
                content=[UserMessageTextContent(text=text)],
                inference_options=InferenceOptions(),
            )
            await store.add_thread_item(thread.id, message, context)
            item_ids.add(message.id)
        _require(len(item_ids) == 3, "Item ID generation returned a duplicate")

        async def load_page(
            after: str | None, limit: int, order: str
        ) -> Page[ThreadItem]:
            return await peer.load_thread_items(thread.id, after, limit, order, context)

        ascending_ids = await _check_item_pagination(load_page, item_ids)

        form = StructuredInputItem(
            id=store.generate_item_id("task", thread, context),
            thread_id=thread.id,
            created_at=created_at + timedelta(seconds=1),
            inputs=[],
        )
        tool = ClientToolCallItem(
            id=store.generate_item_id("tool_call", thread, context),
            thread_id=thread.id,
            created_at=created_at + timedelta(seconds=2),
            call_id="call_store_check",
            name="store_check",
            arguments={},
        )
        claims: tuple[tuple[PendingTurnItem, PendingTurnItem, PendingTurnItem], ...] = (
            (
                form,
                form.model_copy(update={"status": "answered"}),
                form.model_copy(update={"status": "skipped"}),
            ),
            (
                tool,
                tool.model_copy(update={"status": "completed", "output": "one"}),
                tool.model_copy(update={"status": "completed", "output": "two"}),
            ),
        )
        for pending, first, second in claims:
            await store.add_thread_item(thread.id, pending, context)
            outcomes = await asyncio.gather(
                store.claim_pending_item(thread.id, pending, first, context),
                peer.claim_pending_item(thread.id, pending, second, context),
                return_exceptions=True,
            )
            _require(
                outcomes.count(None) == 1
                and sum(isinstance(outcome, ConflictError) for outcome in outcomes)
                == 1,
                f"A pending {pending.type} must have exactly one claimant and one conflict",
            )
            winner = (first, second)[outcomes.index(None)]
            saved = await peer.load_item(thread.id, pending.id, context)
            _require(saved == winner, f"The winning {pending.type} claim must be saved")

        target, removed = await store.truncate_after(
            thread.id, ascending_ids[0], context
        )
        remaining = await load_page(None, 10, "asc")
        _require(
            target.id == ascending_ids[0]
            and removed == [tool.id, form.id, *ascending_ids[:0:-1]]
            and [item.id for item in remaining.data] == [target.id],
            "Retry must remove every item after the target in newest-first order",
        )
        hidden = SDKHiddenContextItem(
            id=store.generate_item_id("sdk_hidden_context", thread, context),
            thread_id=thread.id,
            created_at=created_at + timedelta(seconds=3),
            content="Internal store check",
        )
        await store.add_thread_item(thread.id, hidden, context)
        visible = await peer.load_thread_items(
            thread.id, None, 1, "desc", context, include_hidden=False
        )
        _require(
            [item.id for item in visible.data] == [target.id]
            and not visible.has_more
            and visible.after is None,
            "Visible history must exclude hidden items before applying the page limit",
        )
    finally:
        await store.delete_thread(thread.id, context)

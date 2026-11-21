"""Checks for application implementations of :class:`ThreadStore`."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .models import (
    ClientToolCallItem,
    HiddenContextItem,
    InferenceOptions,
    SDKHiddenContextItem,
    StructuredInputItem,
    ThreadMetadata,
    UserMessageItem,
    UserMessageTextContent,
)
from .store import ConflictError, NotFoundError, PendingTurnItem, ThreadStore
from .turns import ClientToolInvocation, TurnOutcome, TurnState


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _check_item_pagination[TContext](
    store: ThreadStore[TContext],
    thread_id: str,
    context: TContext,
    stored_ids: list[str],
    *,
    exclude_ids: frozenset[str] = frozenset(),
) -> None:
    scope = "browser-visible" if exclude_ids else "stored"
    expected = [item_id for item_id in stored_ids if item_id not in exclude_ids]
    for order, ordered_ids in (("asc", expected), ("desc", expected[::-1])):
        cursor = None
        for index, item_id in enumerate(ordered_ids):
            page = await store.load_thread_items(
                thread_id, cursor, 1, order, context, include_hidden=not exclude_ids
            )
            more_items = index < len(ordered_ids) - 1
            _require(
                [item.id for item in page.data] == [item_id]
                and page.has_more == more_items
                and (page.after is not None) == more_items
                and page.after not in exclude_ids,
                f"Unexpected item or cursor in {scope} {order} history at page {index + 1}",
            )
            cursor = page.after


@asynccontextmanager
async def _disposable_thread[TContext](
    store: ThreadStore[TContext], context: TContext
) -> AsyncIterator[ThreadMetadata]:
    thread = ThreadMetadata(
        id=store.generate_thread_id(context),
        created_at=datetime.now(UTC),
        metadata={"kept": True, "remove": True},
    )
    await store.save_thread(thread, context)
    try:
        yield thread
    finally:
        await store.delete_thread(thread.id, context)


async def _race_claims[TContext](
    store: ThreadStore[TContext],
    peer: ThreadStore[TContext],
    context: TContext,
    pending: PendingTurnItem,
    first: PendingTurnItem,
    second: PendingTurnItem,
) -> tuple[str, ClientToolInvocation | None]:
    claim_ids = str(uuid4()), str(uuid4())
    outcomes = await asyncio.gather(
        store.claim_pending_item(
            pending.thread_id, pending, first, context, next_turn_id=claim_ids[0]
        ),
        peer.claim_pending_item(
            pending.thread_id, pending, second, context, next_turn_id=claim_ids[1]
        ),
        return_exceptions=True,
    )
    winners = [
        (claim_id, replacement, outcome)
        for claim_id, replacement, outcome in zip(
            claim_ids, (first, second), outcomes, strict=True
        )
        if not isinstance(outcome, BaseException)
    ]
    _require(
        len(winners) == 1
        and sum(isinstance(outcome, ConflictError) for outcome in outcomes) == 1,
        f"A pending {pending.type} must have exactly one claimant and one conflict",
    )
    ((claim_id, replacement, result),) = winners
    saved = await peer.load_item(pending.thread_id, pending.id, context)
    _require(saved == replacement, f"The winning {pending.type} claim must be saved")
    return claim_id, result


async def _check_retry[TContext](
    store: ThreadStore[TContext],
    peer: ThreadStore[TContext],
    context: TContext,
    thread_id: str,
    target_id: str,
    later_ids: list[str],
) -> None:
    target = await store.truncate_after(thread_id, target_id, context)
    remaining = await peer.load_thread_items(thread_id, None, 10, "asc", context)
    _require(
        target.id == target_id
        and [item.id for item in remaining.data] == [target_id]
        and not remaining.has_more
        and remaining.after is None,
        "Retry must remove every later item from paginated history",
    )
    for item_id in later_ids:
        try:
            await peer.load_item(thread_id, item_id, context)
        except NotFoundError:
            continue
        raise AssertionError("Retry must hide each truncated item from direct reads")


async def check_thread_store[TContext](
    store: ThreadStore[TContext],
    context: TContext,
    *,
    other_context: TContext,
    peer: ThreadStore[TContext] | None = None,
) -> None:
    """Check core thread storage, including form claims, retries, and hidden items.

    Run against a disposable database with contexts for two distinct users.
    Supply an independent store instance sharing that database to check
    concurrent operations across connections. The check writes and deletes
    a synthetic thread. A failed condition raises an informative AssertionError.
    """
    peer = peer if peer is not None else store
    async with _disposable_thread(store, context) as thread:
        created_at = thread.created_at
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

        older_turn, newer_turn = str(uuid4()), str(uuid4())
        await store.begin_turn(thread.id, older_turn, context)
        await store.start_turn(thread.id, older_turn, context)
        await peer.begin_turn(thread.id, newer_turn, context)
        try:
            await store.patch_thread(
                thread.id, {"title": "Stale title"}, context, turn_id=older_turn
            )
        except ConflictError:
            pass
        else:
            raise AssertionError("A superseded turn can still patch the thread")
        _require(
            await peer.load_turn_state(thread.id, older_turn, context)
            == TurnState.SUPERSEDED,
            "The older turn must retain its superseded state",
        )
        _require(
            await store.finish_turn(
                thread.id, newer_turn, TurnOutcome.CANCELLED, context
            )
            == TurnState.ABANDONED,
            "Closing an accepted turn must record that it never started",
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

        ascending = await peer.load_thread_items(thread.id, None, 3, "asc", context)
        ascending_ids = [item.id for item in ascending.data]
        _require(
            len(ascending_ids) == len(item_ids) and set(ascending_ids) == item_ids,
            "Pagination must return each item with a tied timestamp once",
        )

        form = StructuredInputItem(
            id=store.generate_item_id("task", thread, context),
            thread_id=thread.id,
            created_at=created_at + timedelta(seconds=2),
            inputs=[],
        )
        await store.add_thread_item(thread.id, form, context)
        active_turn = str(uuid4())
        await store.begin_turn(thread.id, active_turn, context)
        answered = form.model_copy(update={"status": "answered"})
        for active_state in (TurnState.ACCEPTED, TurnState.RUNNING):
            if active_state == TurnState.RUNNING:
                await store.start_turn(thread.id, active_turn, context)
            try:
                await peer.claim_pending_item(
                    thread.id, form, answered, context, next_turn_id=str(uuid4())
                )
            except ConflictError:
                pass
            else:
                raise AssertionError(f"A form claim replaced the {active_state} turn")
            _require(
                await peer.load_turn_state(thread.id, active_turn, context)
                == active_state
                and await peer.load_item(thread.id, form.id, context) == form,
                "A rejected form claim must preserve the active turn and pending form",
            )
        await store.finish_turn(thread.id, active_turn, TurnOutcome.COMPLETED, context)
        _, result = await _race_claims(
            store,
            peer,
            context,
            form,
            answered,
            form.model_copy(update={"status": "skipped"}),
        )
        _require(result is None, "A form claim must return no private invocation")

        app_note = HiddenContextItem(
            id=str(uuid4()),
            thread_id=thread.id,
            created_at=created_at + timedelta(seconds=1),
            content="App context",
        )
        sdk_note = SDKHiddenContextItem(
            id=store.generate_item_id("sdk_hidden_context", thread, context),
            thread_id=thread.id,
            created_at=created_at + timedelta(seconds=3),
            content="SDK context",
        )
        last_message = UserMessageItem(
            id=store.generate_item_id("message", thread, context),
            thread_id=thread.id,
            created_at=created_at + timedelta(seconds=4),
            content=[UserMessageTextContent(text="Last message")],
            inference_options=InferenceOptions(),
        )
        for item in (app_note, sdk_note, last_message):
            await store.add_thread_item(thread.id, item, context)
        stored_ids = [
            *ascending_ids,
            app_note.id,
            form.id,
            sdk_note.id,
            last_message.id,
        ]
        await _check_item_pagination(peer, thread.id, context, stored_ids)
        await _check_item_pagination(
            peer,
            thread.id,
            context,
            stored_ids,
            exclude_ids=frozenset({app_note.id, sdk_note.id}),
        )
        await _check_retry(
            store,
            peer,
            context,
            thread.id,
            ascending_ids[0],
            stored_ids[1:],
        )


async def check_client_tool_store[TContext](
    store: ThreadStore[TContext],
    context: TContext,
    *,
    other_context: TContext,
    peer: ThreadStore[TContext] | None = None,
) -> None:
    """Check the optional browser-tool store contract using a disposable thread.

    Run this alongside check_thread_store for stores that support browser callbacks.
    Pass two distinct users and, optionally, a second connection to the same database.
    """
    peer = peer if peer is not None else store
    async with _disposable_thread(store, context) as thread:
        anchor = UserMessageItem(
            id=store.generate_item_id("message", thread, context),
            thread_id=thread.id,
            created_at=thread.created_at,
            content=[UserMessageTextContent(text="Start a tool call")],
            inference_options=InferenceOptions(),
        )
        await store.add_thread_item(thread.id, anchor, context)

        async def register_tool() -> tuple[ClientToolCallItem, ClientToolInvocation]:
            owner = str(uuid4())
            await store.begin_turn(thread.id, owner, context)
            await store.start_turn(thread.id, owner, context)
            item = ClientToolCallItem(
                id=store.generate_item_id("tool_call", thread, context),
                thread_id=thread.id,
                created_at=thread.created_at + timedelta(seconds=1),
                name="store_check",
                arguments={},
            )
            invocation = ClientToolInvocation(
                item_id=item.id,
                thread_id=thread.id,
                turn_id=owner,
                call_id=f"call_{owner}",
            )
            await store.register_client_tool_call(thread.id, item, invocation, context)
            return item, invocation

        tool, invocation = await register_tool()
        _require(
            await peer.load_client_tool_status(thread.id, tool.id, context) == "pending"
            and await peer.load_client_tool_invocation(thread.id, tool.id, context)
            == invocation,
            "A pending callback and its private invocation must be readable across store instances",
        )
        try:
            await peer.load_client_tool_invocation(thread.id, tool.id, other_context)
        except NotFoundError:
            pass
        else:
            raise AssertionError("Another user can read the private invocation")

        hidden = SDKHiddenContextItem(
            id=store.generate_item_id("sdk_hidden_context", thread, context),
            thread_id=thread.id,
            created_at=thread.created_at + timedelta(seconds=2),
            content="Internal store check",
        )
        await peer.add_thread_item(thread.id, hidden, context)
        winner_id, claimed = await _race_claims(
            store,
            peer,
            context,
            tool,
            tool.model_copy(update={"status": "completed", "output": "one"}),
            tool.model_copy(update={"status": "completed", "output": "two"}),
        )
        _require(
            claimed == invocation.model_copy(update={"continuation_turn_id": winner_id})
            and await peer.load_client_tool_invocation(thread.id, tool.id, context)
            == claimed,
            "The winning callback must bind the same private invocation and turn",
        )
        _require(
            await peer.load_client_tool_status(thread.id, tool.id, context)
            == "completed",
            "The callback lifecycle must complete with the winning claim",
        )
        cancelled, _ = await register_tool()
        await _check_retry(
            store,
            peer,
            context,
            thread.id,
            anchor.id,
            [tool.id, hidden.id, cancelled.id],
        )
        _require(
            await peer.load_client_tool_status(thread.id, cancelled.id, context)
            == "cancelled",
            "Retry must close active callbacks and retain their lifecycle",
        )

        await store.add_thread_item(thread.id, hidden, context)
        explicitly_cancelled, _ = await register_tool()
        await peer.cancel_client_tool_call(thread.id, explicitly_cancelled.id, context)
        visible = await peer.load_thread_items(
            thread.id, None, 1, "desc", context, include_hidden=False
        )
        _require(
            [item.id for item in visible.data] == [anchor.id]
            and not visible.has_more
            and visible.after is None,
            "Visible history must exclude hidden and cancelled items before applying the page limit",
        )

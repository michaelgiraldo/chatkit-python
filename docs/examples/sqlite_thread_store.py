"""File-backed SQLite example for local ChatKit apps."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel

from chatkit.models import (
    ClientToolCallItem,
    Page,
    ThreadItem,
    ThreadMetadata,
    UserMessageItem,
)
from chatkit.store import (
    HIDDEN_ITEM_TYPES,
    ClientToolCallbackStatus,
    ConflictError,
    NotFoundError,
    PendingTurnItem,
    ThreadPatch,
    ThreadStore,
    TurnSupersededError,
    apply_thread_patch,
    validate_client_tool_registration,
    validate_pending_claim,
)
from chatkit.turns import ClientToolInvocation, TurnOutcome, TurnState


class UserContext(Protocol):
    """Request context supplied by the application."""

    @property
    def user_id(self) -> str: ...


class ThreadData(BaseModel):
    thread: ThreadMetadata


class ItemData(BaseModel):
    item: ThreadItem


def _sort_timestamp(created_at: datetime) -> str:
    # Sort timestamps on the same UTC timeline, including legacy local timestamps.
    return created_at.astimezone(UTC).isoformat(timespec="microseconds")


class SQLiteThreadStore[TContext: UserContext](ThreadStore[TContext]):
    """A durable local store with a database transaction per write."""

    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        self._create_tables()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, uri=True)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_tables(self) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL,
                UNIQUE (id, user_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (thread_id, user_id) REFERENCES threads (id, user_id)
                    ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS thread_turns (
                id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                state TEXT NOT NULL,
                is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
                PRIMARY KEY (thread_id, user_id, id),
                FOREIGN KEY (thread_id, user_id) REFERENCES threads (id, user_id)
                    ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS one_current_turn
                ON thread_turns (thread_id, user_id) WHERE is_current = 1"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS client_tool_callbacks (
                id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'cancelled')),
                turn_id TEXT NOT NULL,
                invocation TEXT NOT NULL,
                PRIMARY KEY (thread_id, user_id, id),
                FOREIGN KEY (thread_id, user_id) REFERENCES threads (id, user_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (thread_id, user_id, turn_id) REFERENCES thread_turns (thread_id, user_id, id)
                    ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS one_pending_client_tool_callback
                ON client_tool_callbacks (thread_id, user_id) WHERE status = 'pending'"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS threads_by_user ON threads (user_id, created_at, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS items_by_thread ON items (thread_id, user_id, created_at, id)"
            )

    async def load_thread(self, thread_id: str, context: TContext) -> ThreadMetadata:
        with closing(self._create_connection()) as conn, conn:
            thread_cursor = conn.execute(
                "SELECT data FROM threads WHERE id = ? AND user_id = ?",
                (thread_id, context.user_id),
            ).fetchone()
            if thread_cursor is None:
                raise NotFoundError(f"Thread {thread_id} not found")

            return ThreadData.model_validate_json(thread_cursor[0]).thread

    async def save_thread(self, thread: ThreadMetadata, context: TContext) -> None:
        with closing(self._create_connection()) as conn, conn:
            thread_data = ThreadData(thread=thread)
            saved = conn.execute(
                """INSERT INTO threads (id, user_id, created_at, data) VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created_at = excluded.created_at, data = excluded.data
                WHERE threads.user_id = excluded.user_id""",
                (
                    thread.id,
                    context.user_id,
                    _sort_timestamp(thread.created_at),
                    thread_data.model_dump_json(),
                ),
            )
            if not saved.rowcount:
                raise NotFoundError(f"Thread {thread.id} not found")

    async def begin_turn(
        self,
        thread_id: str,
        turn_id: str,
        context: TContext,
        *,
        message: UserMessageItem | None = None,
        create_thread: ThreadMetadata | None = None,
    ) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id)
            if create_thread is None:
                self._require_thread(conn, *scope)
            else:
                if create_thread.id != thread_id:
                    raise ValueError("New thread must use the requested thread ID")
                inserted = conn.execute(
                    """INSERT INTO threads (id, user_id, created_at, data)
                    VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING""",
                    (
                        *scope,
                        _sort_timestamp(create_thread.created_at),
                        ThreadData(thread=create_thread).model_dump_json(),
                    ),
                )
                if not inserted.rowcount:
                    raise ConflictError("The thread already exists")
            self._replace_current_turn(conn, *scope, turn_id)
            if message is not None:
                self._insert_item(conn, *scope, message)

    def _replace_current_turn(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        user_id: str,
        turn_id: str,
        *,
        continuing: bool = False,
    ) -> None:
        scope = (thread_id, user_id)
        conn.execute(
            """UPDATE thread_turns SET is_current = 0, state = CASE
                WHEN state IN (?, ?, ?) THEN ? ELSE state END
            WHERE thread_id = ? AND user_id = ? AND is_current = 1""",
            (
                TurnState.ACCEPTED,
                TurnState.RUNNING,
                TurnState.WAITING,
                TurnState.WAITING if continuing else TurnState.SUPERSEDED,
                *scope,
            ),
        )
        inserted = conn.execute(
            """INSERT INTO thread_turns (thread_id, user_id, id, state, is_current)
            VALUES (?, ?, ?, ?, 1) ON CONFLICT(thread_id, user_id, id) DO NOTHING""",
            (*scope, turn_id, TurnState.ACCEPTED),
        )
        if not inserted.rowcount:
            raise ConflictError("The turn ID is already reserved")
        self._cancel_pending_callbacks(conn, *scope)

    @staticmethod
    def _load_turn(
        conn: sqlite3.Connection, thread_id: str, user_id: str, turn_id: str
    ) -> tuple[TurnState, bool] | None:
        row = conn.execute(
            """SELECT state, is_current FROM thread_turns
            WHERE thread_id = ? AND user_id = ? AND id = ?""",
            (thread_id, user_id, turn_id),
        ).fetchone()
        return (TurnState(row[0]), bool(row[1])) if row is not None else None

    def _require_current_turn(
        self, conn: sqlite3.Connection, thread_id: str, user_id: str, turn_id: str
    ) -> TurnState:
        turn = self._load_turn(conn, thread_id, user_id, turn_id)
        if turn is None or not turn[1]:
            raise TurnSupersededError("The turn no longer owns this thread")
        return turn[0]

    def _require_running_turn(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        user_id: str,
        turn_id: str | None,
    ) -> None:
        if (
            turn_id is not None
            and self._require_current_turn(conn, thread_id, user_id, turn_id)
            != TurnState.RUNNING
        ):
            raise ConflictError("Only a running turn can change the thread")

    @staticmethod
    def _write_turn_state(
        conn: sqlite3.Connection, scope: tuple[str, str, str], state: TurnState
    ) -> None:
        conn.execute(
            """UPDATE thread_turns SET state = ?
            WHERE thread_id = ? AND user_id = ? AND id = ?""",
            (state, *scope),
        )

    async def start_turn(self, thread_id: str, turn_id: str, context: TContext) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id, turn_id)
            if self._require_current_turn(conn, *scope) != TurnState.ACCEPTED:
                raise ConflictError("Only an accepted turn can start")
            self._write_turn_state(conn, scope, TurnState.RUNNING)

    async def require_current_turn(
        self, thread_id: str, turn_id: str, context: TContext
    ) -> None:
        with closing(self._create_connection()) as conn:
            self._require_current_turn(conn, thread_id, context.user_id, turn_id)

    async def load_turn_state(
        self, thread_id: str, turn_id: str, context: TContext
    ) -> TurnState:
        with closing(self._create_connection()) as conn:
            turn = self._load_turn(conn, thread_id, context.user_id, turn_id)
            if turn is None:
                raise NotFoundError("Turn not found")
            return turn[0]

    async def finish_turn(
        self, thread_id: str, turn_id: str, outcome: TurnOutcome, context: TContext
    ) -> TurnState:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id, turn_id)
            turn = self._load_turn(conn, *scope)
            if turn is None:
                raise NotFoundError("Turn not found")
            state, is_current = turn
            if not is_current:
                return state
            if state == TurnState.RUNNING:
                state = TurnState(outcome.value)
            elif state == TurnState.ACCEPTED and outcome == TurnOutcome.CANCELLED:
                state = TurnState.ABANDONED
            else:
                return state
            if state != TurnState.WAITING:
                self._cancel_pending_callbacks(
                    conn, thread_id, context.user_id, turn_id=turn_id
                )
            self._write_turn_state(conn, scope, state)
            return state

    async def patch_thread(
        self,
        thread_id: str,
        patch: ThreadPatch,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> ThreadMetadata:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_running_turn(conn, thread_id, context.user_id, turn_id)
            row = conn.execute(
                "SELECT data FROM threads WHERE id = ? AND user_id = ?",
                (thread_id, context.user_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Thread {thread_id} not found")
            current = ThreadData.model_validate_json(row[0]).thread
            updated = apply_thread_patch(current, patch)
            conn.execute(
                "UPDATE threads SET data = ? WHERE id = ? AND user_id = ?",
                (
                    ThreadData(thread=updated).model_dump_json(),
                    thread_id,
                    context.user_id,
                ),
            )
        return updated

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
        if order not in ("asc", "desc"):
            raise ValueError(f"Unsupported sort order: {order}")
        if limit < 1:
            raise ValueError("Page limit must be positive")
        with closing(self._create_connection()) as conn, conn:
            condition = "thread_id = ? AND user_id = ?"
            params: list[str | int | None] = [thread_id, context.user_id]
            if not include_hidden:
                condition += """
                    AND json_extract(data, '$.item.type') NOT IN (?, ?)
                    AND (
                        json_extract(data, '$.item.type') != 'client_tool_call'
                        OR json_extract(data, '$.item.status') = 'completed'
                        OR EXISTS (
                            SELECT 1 FROM client_tool_callbacks AS callback
                            WHERE callback.thread_id = items.thread_id
                                AND callback.user_id = items.user_id
                                AND callback.id = items.id
                                AND callback.status = 'pending'
                        )
                    )
                """
                params.extend(HIDDEN_ITEM_TYPES)
            created_after: str | None = None
            if after is not None:
                cursor_row = conn.execute(
                    f"SELECT created_at FROM items WHERE {condition} AND id = ?",
                    (*params, after),
                ).fetchone()
                if cursor_row is None:
                    raise NotFoundError(f"Item {after} not found")
                created_after = cursor_row[0]

            query = f"SELECT data FROM items WHERE {condition}"
            if created_after is not None:
                comparison = ">" if order == "asc" else "<"
                query += f" AND (created_at, id) {comparison} (?, ?)"
                params.extend((created_after, after))

            query += f" ORDER BY created_at {order}, id {order} LIMIT ?"
            params.append(limit + 1)

            items_cursor = conn.execute(query, params)
            items = [
                ItemData.model_validate_json(item[0]).item for item in items_cursor
            ]
            has_more = len(items) > limit
            next_after: str | None = None
            if has_more:
                items = items[:limit]
                next_after = items[-1].id
            return Page[ThreadItem](data=items, has_more=has_more, after=next_after)

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: TContext,
    ) -> Page[ThreadMetadata]:
        if order not in ("asc", "desc"):
            raise ValueError(f"Unsupported sort order: {order}")
        if limit < 1:
            raise ValueError("Page limit must be positive")
        with closing(self._create_connection()) as conn, conn:
            created_after: str | None = None
            if after is not None:
                created_of_after_thread = conn.execute(
                    "SELECT created_at FROM threads WHERE id = ? AND user_id = ?",
                    (after, context.user_id),
                ).fetchone()
                if created_of_after_thread is None:
                    raise NotFoundError(f"Thread {after} not found")
                created_after = created_of_after_thread[0]

            query = """
                SELECT data
                FROM threads
                WHERE user_id = ?
            """
            params: list[str | int | None] = [context.user_id]
            if created_after is not None:
                comparison = ">" if order == "asc" else "<"
                query += f" AND (created_at, id) {comparison} (?, ?)"
                params.extend((created_after, after))
            query += f" ORDER BY created_at {order}, id {order} LIMIT ?"
            params.append(limit + 1)
            threads_cursor = conn.execute(query, params).fetchall()
            result = []
            for data in threads_cursor:
                thread = ThreadData.model_validate_json(data[0]).thread
                result.append(thread)
            next_after = None
            has_more = len(result) > limit
            if has_more:
                result = result[:limit]
                next_after = result[-1].id
            return Page[ThreadMetadata](
                data=result, has_more=has_more, after=next_after
            )

    async def add_thread_item(
        self,
        thread_id: str,
        item: ThreadItem,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        if isinstance(item, ClientToolCallItem):
            raise ConflictError("Client tool items require registration and completion")
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id)
            self._require_running_turn(conn, *scope, turn_id)
            self._require_thread(conn, *scope)
            self._insert_item(conn, *scope, item)

    def _insert_item(
        self,
        conn: sqlite3.Connection,
        thread_id: str,
        user_id: str,
        item: ThreadItem,
    ) -> None:
        if item.thread_id != thread_id:
            raise ValueError("Item must belong to the requested thread")
        scope = (thread_id, user_id)
        if self._callback_status(conn, *scope, item.id) is not None:
            raise ConflictError("The client tool callback ID is already reserved")
        conn.execute(
            "INSERT INTO items (id, thread_id, user_id, created_at, data) VALUES (?, ?, ?, ?, ?)",
            (
                item.id,
                *scope,
                _sort_timestamp(item.created_at),
                ItemData(item=item).model_dump_json(),
            ),
        )
        if isinstance(item, UserMessageItem):
            self._cancel_pending_callbacks(conn, *scope)

    async def register_client_tool_call(
        self,
        thread_id: str,
        item: ClientToolCallItem,
        invocation: ClientToolInvocation,
        context: TContext,
    ) -> None:
        validate_client_tool_registration(thread_id, item, invocation)
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id)
            self._require_running_turn(conn, *scope, invocation.turn_id)
            self._insert_item(conn, *scope, item)
            try:
                conn.execute(
                    """INSERT INTO client_tool_callbacks
                        (thread_id, user_id, id, status, turn_id, invocation)
                    VALUES (?, ?, ?, 'pending', ?, ?)""",
                    (*scope, item.id, invocation.turn_id, invocation.model_dump_json()),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError(
                    "The thread already has a pending client tool callback"
                ) from exc

    async def load_client_tool_invocation(
        self, thread_id: str, item_id: str, context: TContext
    ) -> ClientToolInvocation:
        with closing(self._create_connection()) as conn:
            row = conn.execute(
                """SELECT invocation FROM client_tool_callbacks
                WHERE thread_id = ? AND user_id = ? AND id = ?""",
                (thread_id, context.user_id, item_id),
            ).fetchone()
            if row is None:
                raise NotFoundError("Client tool callback not found")
            return ClientToolInvocation.model_validate_json(row[0])

    async def save_item(
        self,
        thread_id: str,
        item: ThreadItem,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        if item.thread_id != thread_id:
            raise ValueError("Item must belong to the requested thread")
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id)
            self._require_running_turn(conn, *scope, turn_id)
            self._require_thread(conn, *scope)
            current = conn.execute(
                """SELECT json_extract(data, '$.item.type') FROM items
                WHERE thread_id = ? AND user_id = ? AND id = ?""",
                (*scope, item.id),
            ).fetchone()
            if (
                isinstance(item, ClientToolCallItem)
                or (current is not None and current[0] == "client_tool_call")
                or self._callback_status(conn, *scope, item.id) is not None
            ):
                raise ConflictError(
                    "Client tool items require registration and completion"
                )
            saved = conn.execute(
                """
                INSERT INTO items (id, thread_id, user_id, created_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created_at = excluded.created_at,
                    data = excluded.data
                WHERE items.thread_id = excluded.thread_id
                    AND items.user_id = excluded.user_id
                """,
                (
                    item.id,
                    thread_id,
                    context.user_id,
                    _sort_timestamp(item.created_at),
                    ItemData(item=item).model_dump_json(),
                ),
            )
            if not saved.rowcount:
                raise NotFoundError(f"Item {item.id} not found in thread {thread_id}")
            if current is None and isinstance(item, UserMessageItem):
                self._cancel_pending_callbacks(conn, *scope)

    @staticmethod
    def _callback_status(
        conn: sqlite3.Connection, thread_id: str, user_id: str, item_id: str
    ) -> ClientToolCallbackStatus | None:
        row = conn.execute(
            """SELECT status FROM client_tool_callbacks
            WHERE thread_id = ? AND user_id = ? AND id = ?""",
            (thread_id, user_id, item_id),
        ).fetchone()
        return cast(ClientToolCallbackStatus, row[0]) if row is not None else None

    async def load_client_tool_status(
        self, thread_id: str, item_id: str, context: TContext
    ) -> ClientToolCallbackStatus:
        with closing(self._create_connection()) as conn:
            status = self._callback_status(conn, thread_id, context.user_id, item_id)
            if status is None:
                raise NotFoundError("Client tool callback not found")
            return status

    async def cancel_client_tool_call(
        self,
        thread_id: str,
        item_id: str,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_running_turn(conn, thread_id, context.user_id, turn_id)
            self._cancel_pending_callbacks(
                conn, thread_id, context.user_id, item_id, turn_id=turn_id
            )

    @staticmethod
    def _cancel_pending_callbacks(
        conn: sqlite3.Connection,
        thread_id: str,
        user_id: str,
        item_id: str | None = None,
        *,
        turn_id: str | None = None,
    ) -> None:
        conn.execute(
            """UPDATE client_tool_callbacks SET status = 'cancelled'
            WHERE thread_id = ? AND user_id = ? AND status = 'pending'
                AND (? IS NULL OR id = ?) AND (? IS NULL OR turn_id = ?)""",
            (thread_id, user_id, item_id, item_id, turn_id, turn_id),
        )

    async def claim_pending_item(
        self,
        thread_id: str,
        expected: PendingTurnItem,
        replacement: PendingTurnItem,
        context: TContext,
        *,
        next_turn_id: str | None = None,
    ) -> ClientToolInvocation | None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id)
            self._require_thread(conn, *scope)
            params = (expected.id, *scope)
            row = conn.execute(
                "SELECT data FROM items WHERE id = ? AND thread_id = ? AND user_id = ?",
                params,
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    f"Item {expected.id} not found in thread {thread_id}"
                )

            validate_pending_claim(
                ItemData.model_validate_json(row[0]).item, expected, replacement
            )
            invocation = None
            if isinstance(expected, ClientToolCallItem):
                if next_turn_id is None:
                    raise ValueError(
                        "A client tool claim requires a continuation turn ID"
                    )
                claimed = conn.execute(
                    """UPDATE client_tool_callbacks SET status = 'completed',
                        invocation = json_set(invocation, '$.continuation_turn_id', ?)
                    WHERE id = ? AND thread_id = ? AND user_id = ? AND status = 'pending'
                    RETURNING turn_id, invocation""",
                    (next_turn_id, *params),
                ).fetchone()
                if claimed is None:
                    raise ConflictError("The client tool callback is no longer pending")
                if self._require_current_turn(conn, *scope, claimed[0]) not in (
                    TurnState.RUNNING,
                    TurnState.WAITING,
                ):
                    raise ConflictError(
                        "The client tool callback no longer owns this thread"
                    )
                invocation = ClientToolInvocation.model_validate_json(claimed[1])
            conn.execute(
                "UPDATE items SET data = ? WHERE id = ? AND thread_id = ? AND user_id = ?",
                (ItemData(item=replacement).model_dump_json(), *params),
            )
            if next_turn_id is not None:
                self._replace_current_turn(
                    conn,
                    *scope,
                    next_turn_id,
                    continuing=isinstance(expected, ClientToolCallItem),
                )
            return invocation

    async def truncate_after(
        self,
        thread_id: str,
        item_id: str,
        context: TContext,
        *,
        next_turn_id: str | None = None,
    ) -> UserMessageItem:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            scope = (thread_id, context.user_id)
            self._require_thread(conn, *scope)
            row = conn.execute(
                """SELECT data, created_at FROM items
                WHERE id = ? AND thread_id = ? AND user_id = ?""",
                (item_id, *scope),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
            item = ItemData.model_validate_json(row[0]).item
            if not isinstance(item, UserMessageItem):
                raise ConflictError("A retry must start from a user message")
            if next_turn_id is not None:
                self._replace_current_turn(conn, *scope, next_turn_id)
            else:
                self._cancel_pending_callbacks(conn, *scope)
            conn.execute(
                """DELETE FROM items
                WHERE thread_id = ? AND user_id = ? AND (created_at, id) > (?, ?)""",
                (*scope, row[1], item_id),
            )
        return item

    @staticmethod
    def _require_thread(conn: sqlite3.Connection, thread_id: str, user_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM threads WHERE id = ? AND user_id = ?", (thread_id, user_id)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Thread {thread_id} not found")

    async def load_item(
        self, thread_id: str, item_id: str, context: TContext
    ) -> ThreadItem:
        with closing(self._create_connection()) as conn, conn:
            cursor = conn.execute(
                "SELECT data FROM items WHERE id = ? AND thread_id = ? AND user_id = ?",
                (item_id, thread_id, context.user_id),
            ).fetchone()
            if cursor is None:
                raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
            return ItemData.model_validate_json(cursor[0]).item

    async def delete_thread(self, thread_id: str, context: TContext) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute(
                "DELETE FROM threads WHERE id = ? AND user_id = ?",
                (thread_id, context.user_id),
            )

    async def delete_thread_item(
        self,
        thread_id: str,
        item_id: str,
        context: TContext,
        *,
        turn_id: str | None = None,
    ) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_running_turn(conn, thread_id, context.user_id, turn_id)
            self._cancel_pending_callbacks(conn, thread_id, context.user_id, item_id)
            conn.execute(
                "DELETE FROM items WHERE id = ? AND thread_id = ? AND user_id = ?",
                (item_id, thread_id, context.user_id),
            )

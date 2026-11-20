from contextlib import closing
from pathlib import Path

from pydantic import BaseModel

from chatkit.models import Attachment
from chatkit.store import AttachmentMetadataStore, NotFoundError
from docs.examples.sqlite_thread_store import SQLiteThreadStore
from tests._types import RequestContext


class AttachmentData(BaseModel):
    attachment: Attachment


class SQLiteStore(
    SQLiteThreadStore[RequestContext], AttachmentMetadataStore[RequestContext]
):
    def __init__(self, db_path: str | Path):
        super().__init__(db_path)
        with closing(self._create_connection()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                data TEXT NOT NULL
                )"""
            )

    async def save_attachment(
        self, attachment: Attachment, context: RequestContext
    ) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute(
                """
                INSERT INTO files (id, user_id, data) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET data = excluded.data
                WHERE files.user_id = excluded.user_id
                """,
                (
                    attachment.id,
                    context.user_id,
                    AttachmentData(attachment=attachment).model_dump_json(),
                ),
            )
            conn.commit()

    async def load_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> Attachment:
        with closing(self._create_connection()) as conn, conn:
            file_cursor = conn.execute(
                "SELECT data FROM files WHERE id = ? AND user_id = ?",
                (attachment_id, context.user_id),
            ).fetchone()
            if file_cursor is None:
                raise NotFoundError(f"File {attachment_id} not found")
            return AttachmentData.model_validate_json(file_cursor[0]).attachment

    async def delete_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> None:
        with closing(self._create_connection()) as conn, conn:
            conn.execute(
                "DELETE FROM files WHERE id = ? AND user_id = ?",
                (attachment_id, context.user_id),
            )
            conn.commit()

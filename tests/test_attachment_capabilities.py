from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from chatkit.server import ChatKitServer, NonStreamingResult, StreamingResult
from chatkit.store import AttachmentStore, NotFoundError
from chatkit.turns import Turn
from chatkit.types import (
    Attachment,
    AttachmentCreateParams,
    AttachmentDeleteParams,
    AttachmentsCreateReq,
    AttachmentsDeleteReq,
    FileAttachment,
    InferenceOptions,
    ThreadCreatedEvent,
    ThreadCreateParams,
    ThreadsCreateReq,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageItem,
    UserMessageTextContent,
)
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.server import DEFAULT_CONTEXT, decode_streaming_result


class UploadGateway(AttachmentStore[RequestContext]):
    def __init__(self):
        self.files: set[str] = set()

    async def create_attachment(
        self, input: AttachmentCreateParams, context: RequestContext
    ) -> Attachment:
        self.files.add("file-example")
        return FileAttachment(
            id="file-example", mime_type=input.mime_type, name=input.name
        )

    async def delete_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> None:
        self.files.remove(attachment_id)


class AttachmentTestServer(ChatKitServer[RequestContext]):
    async def respond(
        self, turn: Turn[RequestContext]
    ) -> AsyncIterator[ThreadStreamEvent]:
        return
        yield


CREATE_ATTACHMENT = AttachmentsCreateReq(
    params=AttachmentCreateParams(name="example.txt", size=7, mime_type="text/plain")
)
DELETE_ATTACHMENT = AttachmentsDeleteReq(
    params=AttachmentDeleteParams(attachment_id="file-example")
)
CREATE_THREAD = ThreadsCreateReq(
    params=ThreadCreateParams(
        input=UserMessageInput(
            content=[UserMessageTextContent(text="Here is the file.")],
            attachments=["file-example"],
            inference_options=InferenceOptions(),
        )
    )
)


async def test_attachments_use_explicit_metadata_store(tmp_path: Path):
    threads = SQLiteStore(str(tmp_path / "threads.sqlite"))
    metadata = SQLiteStore(str(tmp_path / "metadata.sqlite"))
    uploads = UploadGateway()
    server = AttachmentTestServer(
        threads, attachment_store=uploads, attachment_metadata_store=metadata
    )

    result = await server.process(CREATE_ATTACHMENT.model_dump_json(), DEFAULT_CONTEXT)
    assert isinstance(result, NonStreamingResult)
    attachment = FileAttachment.model_validate_json(result.json)
    assert uploads.files == {attachment.id}
    assert await metadata.load_attachment(attachment.id, DEFAULT_CONTEXT) == attachment
    with pytest.raises(NotFoundError):
        await threads.load_attachment(attachment.id, DEFAULT_CONTEXT)

    stream = await server.process(CREATE_THREAD.model_dump_json(), DEFAULT_CONTEXT)
    assert isinstance(stream, StreamingResult)
    events = await decode_streaming_result(stream)
    thread = next(
        event.thread for event in events if isinstance(event, ThreadCreatedEvent)
    )
    saved_attachment = await metadata.load_attachment(attachment.id, DEFAULT_CONTEXT)
    assert saved_attachment.thread_id == thread.id
    items = await threads.load_thread_items(thread.id, None, 10, "asc", DEFAULT_CONTEXT)
    (item,) = items.data
    assert isinstance(item, UserMessageItem)
    assert item.attachments == [saved_attachment]
    assert (await metadata.load_threads(10, None, "asc", DEFAULT_CONTEXT)).data == []

    result = await server.process(DELETE_ATTACHMENT.model_dump_json(), DEFAULT_CONTEXT)
    assert isinstance(result, NonStreamingResult)
    assert uploads.files == set()
    with pytest.raises(NotFoundError):
        await metadata.load_attachment(attachment.id, DEFAULT_CONTEXT)


async def test_missing_attachment_metadata_store_fails_before_writes(tmp_path: Path):
    threads = SQLiteStore(str(tmp_path / "threads.sqlite"))
    uploads = UploadGateway()
    server = AttachmentTestServer(threads, attachment_store=uploads)

    with pytest.raises(RuntimeError, match="AttachmentMetadataStore is not configured"):
        await server.process(CREATE_ATTACHMENT.model_dump_json(), DEFAULT_CONTEXT)
    assert uploads.files == set()

    uploads.files.add("file-example")
    with pytest.raises(RuntimeError, match="AttachmentMetadataStore is not configured"):
        await server.process(DELETE_ATTACHMENT.model_dump_json(), DEFAULT_CONTEXT)
    assert uploads.files == {"file-example"}

    stream = await server.process(CREATE_THREAD.model_dump_json(), DEFAULT_CONTEXT)
    assert isinstance(stream, StreamingResult)
    with pytest.raises(RuntimeError, match="AttachmentMetadataStore is not configured"):
        await anext(stream.json_events)
    assert (await threads.load_threads(10, None, "asc", DEFAULT_CONTEXT)).data == []

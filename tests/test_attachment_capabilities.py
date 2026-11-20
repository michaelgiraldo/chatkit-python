from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chatkit.errors import CustomStreamError, InvalidOperationError
from chatkit.events import (
    ErrorEvent,
    ThreadCreatedEvent,
    ThreadItemDoneEvent,
    ThreadStreamEvent,
)
from chatkit.models import (
    Attachment,
    FileAttachment,
    InferenceOptions,
    ThreadMetadata,
    UserMessageInput,
    UserMessageItem,
    UserMessageTextContent,
)
from chatkit.requests import (
    AttachmentCreateParams,
    AttachmentDeleteParams,
    AttachmentsCreateReq,
    AttachmentsDeleteReq,
    ThreadAddUserMessageParams,
    ThreadCreateParams,
    ThreadRetryAfterItemParams,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
    ThreadsRetryAfterItemReq,
)
from chatkit.server import ChatKitServer, EmptyResult, StreamingResult
from chatkit.store import AttachmentStore, NotFoundError
from chatkit.turns import Turn
from tests._types import RequestContext
from tests.helpers.mock_store import SQLiteStore
from tests.helpers.server import DEFAULT_CONTEXT


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

    result = await server.process(CREATE_ATTACHMENT, DEFAULT_CONTEXT)
    assert isinstance(result, FileAttachment)
    attachment = result
    assert uploads.files == {attachment.id}
    assert await metadata.load_attachment(attachment.id, DEFAULT_CONTEXT) == attachment
    with pytest.raises(NotFoundError):
        await threads.load_attachment(attachment.id, DEFAULT_CONTEXT)

    stream = await server.process(CREATE_THREAD, DEFAULT_CONTEXT)
    assert isinstance(stream, StreamingResult)
    events = [event async for event in stream]
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

    result = await server.process(DELETE_ATTACHMENT, DEFAULT_CONTEXT)
    assert isinstance(result, EmptyResult)
    assert uploads.files == set()
    with pytest.raises(NotFoundError):
        await metadata.load_attachment(attachment.id, DEFAULT_CONTEXT)


@pytest.mark.parametrize("existing_thread", [False, True])
async def test_prepare_user_message_persists_resolved_changes(
    tmp_path: Path, existing_thread: bool
):
    store = SQLiteStore(tmp_path / "prepare.sqlite")
    prepared_messages: list[UserMessageItem] = []

    class PreparingServer(AttachmentTestServer):
        async def prepare_user_message(
            self, thread: ThreadMetadata, item: UserMessageItem, context: RequestContext
        ) -> UserMessageItem:
            assert context is DEFAULT_CONTEXT
            assert thread.id == item.thread_id
            (attachment,) = item.attachments
            assert attachment.thread_id == thread.id
            saved = await store.load_attachment(attachment.id, context)
            assert saved.thread_id is None and saved.metadata is None
            if existing_thread:
                items = await store.load_thread_items(
                    thread.id, None, 10, "asc", context
                )
                assert not items.data
            else:
                with pytest.raises(NotFoundError):
                    await store.load_thread(thread.id, context)

            prepared = item.model_copy(
                update={
                    "attachments": [
                        attachment.model_copy(update={"metadata": {"verified": True}})
                    ],
                    "content": [UserMessageTextContent(text="Prepared message")],
                }
            )
            prepared_messages.append(prepared)
            return prepared

    server = PreparingServer(
        store, attachment_store=UploadGateway(), attachment_metadata_store=store
    )
    await server.process(CREATE_ATTACHMENT, DEFAULT_CONTEXT)
    request: ThreadsCreateReq | ThreadsAddUserMessageReq = CREATE_THREAD
    if existing_thread:
        await store.save_thread(
            ThreadMetadata(id="existing", created_at=datetime.now(UTC)), DEFAULT_CONTEXT
        )
        request = ThreadsAddUserMessageReq(
            params=ThreadAddUserMessageParams(
                thread_id="existing", input=CREATE_THREAD.params.input
            )
        )

    stream = await server.process(request, DEFAULT_CONTEXT)
    assert isinstance(stream, StreamingResult)
    events = [event async for event in stream]
    (prepared,) = prepared_messages
    assert ThreadItemDoneEvent(item=prepared) in events
    assert (
        await store.load_item(prepared.thread_id, prepared.id, DEFAULT_CONTEXT)
        == prepared
    )
    assert prepared.attachments == [
        await store.load_attachment("file-example", DEFAULT_CONTEXT)
    ]

    retry = await server.process(
        ThreadsRetryAfterItemReq(
            params=ThreadRetryAfterItemParams(
                thread_id=prepared.thread_id, item_id=prepared.id
            )
        ),
        DEFAULT_CONTEXT,
    )
    assert isinstance(retry, StreamingResult)
    async for _ in retry:
        pass
    assert prepared_messages == [prepared]


@pytest.mark.parametrize("existing_thread", [False, True])
@pytest.mark.parametrize("stream_error", [False, True])
async def test_prepare_user_message_rejects_before_writes(
    tmp_path: Path, existing_thread: bool, stream_error: bool
):
    store = SQLiteStore(tmp_path / "rejection.sqlite")

    class RejectingServer(AttachmentTestServer):
        async def prepare_user_message(
            self, thread: ThreadMetadata, item: UserMessageItem, context: RequestContext
        ) -> UserMessageItem:
            if stream_error:
                raise CustomStreamError("The file cannot be used.", allow_retry=True)
            raise InvalidOperationError("The file cannot be used.")

    server = RejectingServer(
        store, attachment_store=UploadGateway(), attachment_metadata_store=store
    )
    attachment = await server.process(CREATE_ATTACHMENT, DEFAULT_CONTEXT)
    request: ThreadsCreateReq | ThreadsAddUserMessageReq = CREATE_THREAD
    if existing_thread:
        await store.save_thread(
            ThreadMetadata(id="existing", created_at=datetime.now(UTC)), DEFAULT_CONTEXT
        )
        request = ThreadsAddUserMessageReq(
            params=ThreadAddUserMessageParams(
                thread_id="existing", input=CREATE_THREAD.params.input
            )
        )

    if stream_error:
        stream = await server.process(request, DEFAULT_CONTEXT)
        assert isinstance(stream, StreamingResult)
        assert [event async for event in stream] == [
            ErrorEvent(message="The file cannot be used.", allow_retry=True)
        ]
    else:
        with pytest.raises(InvalidOperationError, match="The file cannot be used."):
            await server.process(request, DEFAULT_CONTEXT)

    assert await store.load_attachment("file-example", DEFAULT_CONTEXT) == attachment
    threads = await store.load_threads(10, None, "asc", DEFAULT_CONTEXT)
    assert [thread.id for thread in threads.data] == (
        ["existing"] if existing_thread else []
    )
    if existing_thread:
        items = await store.load_thread_items(
            "existing", None, 10, "asc", DEFAULT_CONTEXT
        )
        assert not items.data


async def test_missing_attachment_metadata_store_fails_before_writes(tmp_path: Path):
    threads = SQLiteStore(str(tmp_path / "threads.sqlite"))
    uploads = UploadGateway()
    server = AttachmentTestServer(threads, attachment_store=uploads)

    with pytest.raises(RuntimeError, match="AttachmentMetadataStore is not configured"):
        await server.process(CREATE_ATTACHMENT, DEFAULT_CONTEXT)
    assert uploads.files == set()

    uploads.files.add("file-example")
    with pytest.raises(RuntimeError, match="AttachmentMetadataStore is not configured"):
        await server.process(DELETE_ATTACHMENT, DEFAULT_CONTEXT)
    assert uploads.files == {"file-example"}

    with pytest.raises(RuntimeError, match="AttachmentMetadataStore is not configured"):
        await server.process(CREATE_THREAD, DEFAULT_CONTEXT)
    assert (await threads.load_threads(10, None, "asc", DEFAULT_CONTEXT)).data == []

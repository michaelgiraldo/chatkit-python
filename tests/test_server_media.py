import base64
import json
from typing import Any

import pytest
from pydantic import AnyUrl

from chatkit.models import (
    Attachment,
    AttachmentUploadDescriptor,
    AudioInput,
    FileAttachment,
    ImageAttachment,
    InferenceOptions,
    TranscriptionResult,
    UserMessageInput,
    UserMessageTextContent,
)
from chatkit.protocol import encode_message
from chatkit.requests import (
    AttachmentCreateParams,
    AttachmentDeleteParams,
    AttachmentsCreateReq,
    AttachmentsDeleteReq,
    InputTranscribeParams,
    InputTranscribeReq,
    ItemsListParams,
    ItemsListReq,
    ThreadAddUserMessageParams,
    ThreadCreateParams,
    ThreadsAddUserMessageReq,
    ThreadsCreateReq,
)
from chatkit.store import (
    AttachmentStore,
)
from tests._types import RequestContext
from tests.helpers.server import (
    DEFAULT_CONTEXT,
    make_server,
)


class InMemoryFileStore(AttachmentStore):
    def __init__(self):
        self.files = {}

    async def delete_attachment(self, attachment_id: str, context: Any) -> None:
        del self.files[attachment_id]

    async def create_attachment(
        self, input: AttachmentCreateParams, context: RequestContext
    ) -> Attachment:
        # Simple in-memory attachment creation
        id = f"atc_{len(self.files) + 1}"
        metadata = {"source": "test"}
        if input.mime_type.startswith("image/"):
            attachment = ImageAttachment(
                id=id,
                mime_type=input.mime_type,
                name=input.name,
                preview_url=AnyUrl(f"https://example.com/{id}/preview"),
                upload_descriptor=AttachmentUploadDescriptor(
                    url=AnyUrl(f"https://example.com/{id}/upload"),
                    method="PUT",
                    headers={"X-My-Header": "my-value"},
                ),
                metadata=metadata,
            )
        else:
            attachment = FileAttachment(
                id=id,
                mime_type=input.mime_type,
                name=input.name,
                upload_descriptor=AttachmentUploadDescriptor(
                    url=AnyUrl(f"https://example.com/{id}/upload"),
                    method="PUT",
                    headers={"X-My-Header": "my-value"},
                ),
                metadata=metadata,
            )
        self.files[attachment.id] = attachment
        return attachment


async def test_create_file():
    store = InMemoryFileStore()
    file_name = "test-file-name"
    file_content_type = "text/plain"

    with make_server(attachment_store=store) as server:
        attachment = await server.process_non_streaming(
            AttachmentsCreateReq(
                params=AttachmentCreateParams(
                    name=file_name, size=1024, mime_type=file_content_type
                )
            )
        )
        assert isinstance(attachment, FileAttachment)

        # Verify response includes file
        assert attachment.id is not None
        assert attachment.mime_type == file_content_type
        assert attachment.name == file_name
        assert attachment.type == "file"
        assert attachment.upload_descriptor is not None
        assert attachment.upload_descriptor.url == AnyUrl(
            f"https://example.com/{attachment.id}/upload"
        )
        assert attachment.upload_descriptor.method == "PUT"
        assert attachment.upload_descriptor.headers == {"X-My-Header": "my-value"}
        assert attachment.id in store.files

        # The `metadata` field is excluded from serialization by default but still stored server-side
        assert "metadata" not in json.loads(encode_message(attachment))
        assert store.files[attachment.id].metadata == {"source": "test"}


async def test_create_image_file():
    store = InMemoryFileStore()
    file_name = "test-file-name"
    file_content_type = "image/png"

    with make_server(attachment_store=store) as server:
        attachment = await server.process_non_streaming(
            AttachmentsCreateReq(
                params=AttachmentCreateParams(
                    name=file_name,
                    size=1024,
                    mime_type=file_content_type,
                )
            )
        )
        assert isinstance(attachment, ImageAttachment)

        assert attachment.id is not None
        assert attachment.mime_type == file_content_type
        assert attachment.name == file_name
        assert attachment.type == "image"
        assert attachment.preview_url == AnyUrl(
            f"https://example.com/{attachment.id}/preview"
        )
        assert attachment.upload_descriptor is not None
        assert attachment.upload_descriptor.url == AnyUrl(
            f"https://example.com/{attachment.id}/upload"
        )
        assert attachment.upload_descriptor.method == "PUT"
        assert attachment.upload_descriptor.headers == {"X-My-Header": "my-value"}

        assert attachment.id in store.files


async def test_user_message_attachment_metadata_not_serialized():
    attachment = FileAttachment(
        id="file_with_metadata",
        type="file",
        mime_type="text/plain",
        name="test.txt",
        metadata={"source": "test"},
    )

    with make_server() as server:
        metadata_store = server.attachment_metadata_store
        assert metadata_store is not None
        await metadata_store.save_attachment(attachment, DEFAULT_CONTEXT)
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Message with file")],
                        attachments=[attachment.id],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        list_result = await server.process_non_streaming(
            ItemsListReq(params=ItemsListParams(thread_id=thread.id))
        )
        items = json.loads(encode_message(list_result))["data"]
        user_message = next(item for item in items if item["type"] == "user_message")
        attachments = user_message["attachments"]
        assert attachments
        assert all("metadata" not in attachment for attachment in attachments)


async def test_user_message_attachment_thread_id_persisted_on_thread_create():
    attachment = FileAttachment(
        id="file_thread_id_on_create",
        type="file",
        mime_type="text/plain",
        name="test.txt",
        thread_id=None,
    )

    with make_server() as server:
        # Attachment exists before a thread exists, so it may not have a thread_id yet.
        metadata_store = server.attachment_metadata_store
        assert metadata_store is not None
        await metadata_store.save_attachment(attachment, DEFAULT_CONTEXT)

        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Message with file")],
                        attachments=[attachment.id],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(e.thread for e in events if e.type == "thread.created")

        # After the user message is saved, the attachment record should be updated
        # to reflect its thread association.
        stored = await metadata_store.load_attachment(attachment.id, DEFAULT_CONTEXT)
        assert stored.thread_id == thread.id


async def test_user_message_attachment_thread_id_persisted_on_add_user_message():
    with make_server() as server:
        # Create a thread first.
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Start")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(e.thread for e in events if e.type == "thread.created")

        attachment = FileAttachment(
            id="file_thread_id_on_add",
            type="file",
            mime_type="text/plain",
            name="test.txt",
            thread_id=None,
        )
        metadata_store = server.attachment_metadata_store
        assert metadata_store is not None
        await metadata_store.save_attachment(attachment, DEFAULT_CONTEXT)

        # Attach the pre-created attachment to a message in an existing thread.
        await server.process_streaming(
            ThreadsAddUserMessageReq(
                params=ThreadAddUserMessageParams(
                    thread_id=thread.id,
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Second")],
                        attachments=[attachment.id],
                        inference_options=InferenceOptions(),
                    ),
                )
            )
        )

        stored = await metadata_store.load_attachment(attachment.id, DEFAULT_CONTEXT)
        assert stored.thread_id == thread.id


async def test_create_file_without_filestore():
    with pytest.raises(RuntimeError):
        with make_server() as server:
            await server.process_non_streaming(
                AttachmentsCreateReq(
                    params=AttachmentCreateParams(
                        name="test-file-name",
                        size=1024,
                        mime_type="text/plain",
                    )
                )
            )


async def test_delete_file():
    store = InMemoryFileStore()
    store.files["test-file-id"] = {"id": "test-file-id"}
    with make_server(attachment_store=store) as server:
        await server.process_non_streaming(
            AttachmentsDeleteReq(
                params=AttachmentDeleteParams(attachment_id="test-file-id")
            )
        )
        assert store.files == {}


async def test_delete_file_without_filestore():
    with pytest.raises(RuntimeError):
        with make_server() as server:
            await server.process_non_streaming(
                AttachmentsDeleteReq(
                    params=AttachmentDeleteParams(attachment_id="test-file-id")
                )
            )


async def test_input_transcribe_decodes_base64_and_normalizes_mime_type():
    audio_bytes = b"hello audio"
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    seen: dict[str, Any] = {}

    def transcribe_callback(
        audio_input: AudioInput, context: Any
    ) -> TranscriptionResult:
        seen["audio"] = audio_input.data
        seen["mime"] = audio_input.mime_type
        seen["media_type"] = audio_input.media_type
        seen["context"] = context
        return TranscriptionResult(text="ok")

    with make_server(transcribe_callback=transcribe_callback) as server:
        transcription = await server.process_non_streaming(
            InputTranscribeReq(
                params=InputTranscribeParams(
                    audio_base64=audio_b64,
                    mime_type="audio/webm;codecs=opus",
                )
            )
        )
        assert isinstance(transcription, TranscriptionResult)
        assert transcription.text == "ok"

    assert seen["audio"] == audio_bytes
    assert seen["mime"] == "audio/webm;codecs=opus"
    assert seen["media_type"] == "audio/webm"
    assert seen["context"] == DEFAULT_CONTEXT

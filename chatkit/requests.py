"""Request objects decoded from the ChatKit browser protocol."""

from typing import Annotated, Any, Literal, TypeIs

from pydantic import BaseModel, ConfigDict, Field

from . import actions, models


class BaseReq(BaseModel):
    """Base class for all request payloads."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary integration-specific metadata."""


class ThreadsGetByIdReq(BaseReq):
    """Request to fetch a single thread by its identifier."""

    type: Literal["threads.get_by_id"] = "threads.get_by_id"
    params: ThreadGetByIdParams


class ThreadGetByIdParams(BaseModel):
    """Parameters for retrieving a thread by id."""

    thread_id: str


class ThreadsCreateReq(BaseReq):
    """Request to create a new thread from a user message."""

    type: Literal["threads.create"] = "threads.create"
    params: ThreadCreateParams


class ThreadCreateParams(BaseModel):
    """User input required to create a thread."""

    input: models.UserMessageInput


class ThreadListParams(BaseModel):
    """Pagination parameters for listing threads."""

    limit: int | None = None
    order: Literal["asc", "desc"] = "desc"
    after: str | None = None


class ThreadsListReq(BaseReq):
    """Request to list threads."""

    type: Literal["threads.list"] = "threads.list"
    params: ThreadListParams


class ThreadsAddUserMessageReq(BaseReq):
    """Request to append a user message to a thread."""

    type: Literal["threads.add_user_message"] = "threads.add_user_message"
    params: ThreadAddUserMessageParams


class ThreadAddUserMessageParams(BaseModel):
    """Parameters for adding a user message to a thread."""

    input: models.UserMessageInput
    thread_id: str


class ThreadsAddClientToolOutputReq(BaseReq):
    """Request to add a client tool's output to a thread."""

    type: Literal["threads.add_client_tool_output"] = "threads.add_client_tool_output"
    params: ThreadAddClientToolOutputParams


class ClientToolCallbackResult(BaseModel):
    """Output for one exact client tool invocation."""

    model_config = ConfigDict(extra="forbid")

    item_id: Annotated[str, Field(min_length=1, pattern=r"\S")]
    output: Any


class ThreadAddClientToolOutputParams(BaseModel):
    """Parameters for recording tool output in a thread."""

    thread_id: str
    result: ClientToolCallbackResult


class StructuredInputAnswerSubmission(BaseModel):
    """Client-submitted answer for a structured input."""

    values: list[str] = Field(default_factory=list)
    """Text answer values submitted for the structured input."""
    skipped: bool = False
    """Whether this structured input answer was skipped."""


class StructuredInputSubmission(BaseModel):
    """Client-submitted answers for a pending structured input item."""

    status: Literal["answered", "skipped"] = "answered"
    """Overall status for the structured input submission."""
    answers: dict[str, StructuredInputAnswerSubmission] = Field(default_factory=dict)
    """Answers keyed by structured input id."""


class ThreadsAddStructuredInputReq(BaseReq):
    """Request to submit answers for a pending structured input item."""

    type: Literal["threads.add_structured_input"] = "threads.add_structured_input"
    params: ThreadAddStructuredInputParams


class ThreadAddStructuredInputParams(BaseModel):
    """Parameters for adding structured input to a thread."""

    thread_id: str
    item_id: str
    input: StructuredInputSubmission


class ThreadsCustomActionReq(BaseReq):
    """Request to execute a custom action within a thread."""

    type: Literal["threads.custom_action"] = "threads.custom_action"
    params: ThreadCustomActionParams


class ThreadsSyncCustomActionReq(BaseReq):
    """Request to execute a custom action and return a single item update."""

    type: Literal["threads.sync_custom_action"] = "threads.sync_custom_action"
    params: ThreadCustomActionParams


class ThreadCustomActionParams(BaseModel):
    """Parameters describing the custom action to execute."""

    thread_id: str
    item_id: str | None = None
    action: actions.IncomingAction


class ThreadsRetryAfterItemReq(BaseReq):
    """Request to retry processing after a specific thread item."""

    type: Literal["threads.retry_after_item"] = "threads.retry_after_item"
    params: ThreadRetryAfterItemParams


class ThreadRetryAfterItemParams(BaseModel):
    """Parameters specifying which item to retry."""

    thread_id: str
    item_id: str


class ItemsFeedbackReq(BaseReq):
    """Request to submit feedback on specific items."""

    type: Literal["items.feedback"] = "items.feedback"
    params: ItemFeedbackParams


class ItemFeedbackParams(BaseModel):
    """Parameters describing feedback targets and sentiment."""

    thread_id: str
    item_ids: list[str]
    kind: models.FeedbackKind


class AttachmentsDeleteReq(BaseReq):
    """Request to remove an attachment."""

    type: Literal["attachments.delete"] = "attachments.delete"
    params: AttachmentDeleteParams


class AttachmentDeleteParams(BaseModel):
    """Parameters identifying an attachment to delete."""

    attachment_id: str


class AttachmentsCreateReq(BaseReq):
    """Request to register a new attachment."""

    type: Literal["attachments.create"] = "attachments.create"
    params: AttachmentCreateParams


class AttachmentCreateParams(BaseModel):
    """Metadata needed to initialize an attachment."""

    name: str
    size: int
    mime_type: str


class InputTranscribeReq(BaseReq):
    """Request to transcribe an audio payload into text."""

    type: Literal["input.transcribe"] = "input.transcribe"
    params: InputTranscribeParams


class InputTranscribeParams(BaseModel):
    """Parameters for speech transcription."""

    audio_base64: str
    """Base64-encoded audio bytes."""

    mime_type: str
    """Raw MIME type for the audio payload, e.g. "audio/webm;codecs=opus"."""


class ItemsListReq(BaseReq):
    """Request to list items inside a thread."""

    type: Literal["items.list"] = "items.list"
    params: ItemsListParams


class ItemsListParams(BaseModel):
    """Pagination parameters for listing thread items."""

    thread_id: str
    limit: int | None = None
    order: Literal["asc", "desc"] = "desc"
    after: str | None = None


class ThreadsUpdateReq(BaseReq):
    """Request to update thread metadata."""

    type: Literal["threads.update"] = "threads.update"
    params: ThreadUpdateParams


class ThreadUpdateParams(BaseModel):
    """Parameters for updating a thread's properties."""

    thread_id: str
    title: str


class ThreadsDeleteReq(BaseReq):
    """Request to delete a thread."""

    type: Literal["threads.delete"] = "threads.delete"
    params: ThreadDeleteParams


class ThreadDeleteParams(BaseModel):
    """Parameters identifying a thread to delete."""

    thread_id: str


StreamingReq = (
    ThreadsCreateReq
    | ThreadsAddUserMessageReq
    | ThreadsAddClientToolOutputReq
    | ThreadsAddStructuredInputReq
    | ThreadsRetryAfterItemReq
    | ThreadsCustomActionReq
)
"""Union of request types that produce streaming responses."""


NonStreamingReq = (
    ThreadsGetByIdReq
    | ThreadsListReq
    | ItemsListReq
    | ItemsFeedbackReq
    | AttachmentsCreateReq
    | AttachmentsDeleteReq
    | ThreadsUpdateReq
    | ThreadsDeleteReq
    | InputTranscribeReq
    | ThreadsSyncCustomActionReq
)
"""Union of request types that yield immediate responses."""


ChatKitReq = Annotated[
    StreamingReq | NonStreamingReq,
    Field(discriminator="type"),
]
"""Validated browser operations, discriminated by their type field."""


def is_streaming_req(request: ChatKitReq) -> TypeIs[StreamingReq]:
    """Return True if the given request should be processed as streaming."""
    return isinstance(request, StreamingReq)

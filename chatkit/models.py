"""Thread records and domain values stored or passed to application hooks."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AnyUrl, BaseModel, Field, SerializationInfo, model_serializer

from . import icons, widget_models


class Page[T](BaseModel):
    """Paginated collection of records returned from the API."""

    data: list[T] = []
    has_more: bool = False
    after: str | None = None


class AudioInput(BaseModel):
    """Audio input data for transcription."""

    data: bytes
    """Audio data bytes."""

    mime_type: str
    """Raw MIME type for the audio payload, e.g. "audio/webm;codecs=opus"."""

    @property
    def media_type(self) -> str:
        """Media type for the audio payload, e.g. "audio/webm"."""
        return self.mime_type.split(";", 1)[0]


class TranscriptionResult(BaseModel):
    """Input speech transcription result."""

    text: str


### THREAD TYPES


class ThreadMetadata(BaseModel):
    """Metadata describing a thread without its items."""

    title: str | None = None
    id: str
    created_at: datetime
    status: ThreadStatus = Field(default_factory=lambda: ActiveStatus())
    allowed_image_domains: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveStatus(BaseModel):
    """Status indicating the thread is active."""

    type: Literal["active"] = Field(default="active", frozen=True)


class LockedStatus(BaseModel):
    """Status indicating the thread is locked."""

    type: Literal["locked"] = Field(default="locked", frozen=True)
    reason: str | None = None


class ClosedStatus(BaseModel):
    """Status indicating the thread is closed."""

    type: Literal["closed"] = Field(default="closed", frozen=True)
    reason: str | None = None


ThreadStatus = Annotated[
    ActiveStatus | LockedStatus | ClosedStatus,
    Field(discriminator="type"),
]
"""Union of lifecycle states for a thread."""


class Thread(ThreadMetadata):
    """Thread with its paginated items."""

    items: Page[ThreadItem]


### THREAD ITEM TYPES


class ThreadItemBase(BaseModel):
    """Base fields shared by all thread items."""

    id: str
    thread_id: str
    created_at: datetime


class UserMessageItem(ThreadItemBase):
    """Thread item representing a user message."""

    type: Literal["user_message"] = "user_message"
    content: list[UserMessageContent]
    attachments: list[Attachment] = Field(default_factory=list)
    quoted_text: str | None = None
    inference_options: InferenceOptions


class AssistantMessageItem(ThreadItemBase):
    """Thread item representing an assistant message."""

    type: Literal["assistant_message"] = "assistant_message"
    content: list[AssistantMessageContent]


class ClientToolCallItem(ThreadItemBase):
    """Thread item capturing a client tool call."""

    type: Literal["client_tool_call"] = "client_tool_call"
    status: Literal["pending", "completed"] = "pending"
    call_id: str
    name: str
    arguments: dict[str, Any]
    output: Any | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info: SerializationInfo):
        data = serializer(self)
        if self.status == "pending" and (info.context or {}).get("browser_wire"):
            data["arguments"] = {"item_id": self.id, "input": data["arguments"]}
        return data


class WidgetItem(ThreadItemBase):
    """Thread item containing widget content."""

    type: Literal["widget"] = "widget"
    widget: widget_models.WidgetRoot
    copy_text: str | None = None


class GeneratedImage(BaseModel):
    """Generated image."""

    id: str
    url: str


class GeneratedImageItem(ThreadItemBase):
    """Thread item containing a generated image."""

    type: Literal["generated_image"] = "generated_image"
    image: GeneratedImage | None = None


class StructuredInputAnswer(BaseModel):
    """Answer recorded for a structured input."""

    values: list[str] = Field(default_factory=list)
    """Text answer values recorded for the structured input."""
    skipped: bool = False
    """Whether this structured input answer was skipped."""


class StructuredInputBase(BaseModel):
    """Base fields shared by structured inputs."""

    id: str
    """Stable id for this structured input."""
    question: str
    """Question shown to the user."""
    answer: StructuredInputAnswer | None = None
    """Answer recorded for this structured input, if available."""


class StructuredInputMultipleChoiceOption(BaseModel):
    """Option shown for a multiple-choice structured input."""

    value: str
    """Text value submitted when this option is selected."""


class StructuredInputMultipleChoice(StructuredInputBase):
    """Structured input answered by choosing one or more options."""

    type: Literal["multiple_choice"] = "multiple_choice"
    options: list[StructuredInputMultipleChoiceOption]
    """Suggested choices to display before the freeform custom answer affordance."""
    multiple: bool = False
    """Whether the user may submit more than one text value."""


class StructuredInputFreeform(StructuredInputBase):
    """Structured input answered with freeform text."""

    type: Literal["freeform"] = "freeform"
    description: str | None = None
    """Supporting text shown with this input."""


StructuredInput = Annotated[
    StructuredInputMultipleChoice | StructuredInputFreeform,
    Field(discriminator="type"),
]
"""Structured input variants supported by a structured input item."""


class StructuredInputItem(ThreadItemBase):
    """Thread item requesting structured input from the user."""

    type: Literal["structured_input"] = "structured_input"
    status: Literal["pending", "answered", "skipped"] = "pending"
    inputs: list[StructuredInput]


class TaskItem(ThreadItemBase):
    """Thread item containing a task."""

    type: Literal["task"] = "task"
    task: Task


class WorkflowItem(ThreadItemBase):
    """Thread item representing a workflow."""

    type: Literal["workflow"] = "workflow"
    workflow: Workflow


class EndOfTurnItem(ThreadItemBase):
    """Marker item indicating the assistant ends its turn."""

    type: Literal["end_of_turn"] = "end_of_turn"


class HiddenContextItem(ThreadItemBase):
    """
    HiddenContext is never sent to the client. It's not officially part of ChatKit.js.
    It is only used internally to store additional context in a specific place in the thread.
    """

    type: Literal["hidden_context_item"] = "hidden_context_item"
    content: Any


class SDKHiddenContextItem(ThreadItemBase):
    """
    Hidden context that is used by the ChatKit Python SDK for storing additional context
    for internal operations.
    """

    type: Literal["sdk_hidden_context"] = "sdk_hidden_context"
    content: str


ThreadItem = Annotated[
    UserMessageItem
    | AssistantMessageItem
    | ClientToolCallItem
    | WidgetItem
    | GeneratedImageItem
    | StructuredInputItem
    | WorkflowItem
    | TaskItem
    | HiddenContextItem
    | SDKHiddenContextItem
    | EndOfTurnItem,
    Field(discriminator="type"),
]
"""Union of all thread item variants."""


### ASSISTANT MESSAGE TYPES


class AssistantMessageContent(BaseModel):
    """Assistant message content consisting of text and annotations."""

    annotations: list[Annotation] = Field(default_factory=list)
    text: str
    type: Literal["output_text"] = "output_text"


class Annotation(BaseModel):
    """Reference to supporting context attached to assistant output."""

    type: Literal["annotation"] = "annotation"
    source: URLSource | FileSource | EntitySource
    index: int | None = None


### USER MESSAGE TYPES


class UserMessageInput(BaseModel):
    """Payload describing a user message submission."""

    content: list[UserMessageContent]
    attachments: list[str]
    quoted_text: str | None = None
    inference_options: InferenceOptions


class UserMessageTextContent(BaseModel):
    """User message content containing plaintext."""

    type: Literal["input_text"] = "input_text"
    text: str


class UserMessageTagContent(BaseModel):
    """User message content representing an interactive tag."""

    type: Literal["input_tag"] = "input_tag"
    id: str
    text: str
    data: dict[str, Any]
    group: str | None = None
    interactive: bool = False


UserMessageContent = Annotated[
    UserMessageTextContent | UserMessageTagContent, Field(discriminator="type")
]
"""Union of allowed user message content payloads."""


class InferenceOptions(BaseModel):
    """Model and tool configuration for message processing."""

    tool_choice: ToolChoice | None = None
    model: str | None = None


class ToolChoice(BaseModel):
    """Explicit tool selection for the assistant to invoke."""

    id: str


class AttachmentUploadDescriptor(BaseModel):
    """Two-phase upload instructions."""

    url: AnyUrl
    method: Literal["POST", "PUT"]
    """The HTTP method to use when uploading the file for two-phase upload."""
    headers: dict[str, str] = Field(default_factory=dict)
    """Optional headers to include in the upload request."""


class AttachmentBase(BaseModel):
    """Base metadata shared by all attachments."""

    id: str
    name: str
    mime_type: str
    upload_descriptor: AttachmentUploadDescriptor | None = None
    """
    Two-phase upload instructions.
    Should be set to None after upload is complete or when using direct upload
    where uploading happens when creating the attachment object.
    """
    thread_id: str | None = None
    """
    The thread the attachment belongs to, if any.
    Added when the user message that contains the attachment is saved to store.
    """
    metadata: dict[str, Any] | None = None
    """
    Integration-only metadata stored with the attachment. Ignored by ChatKit and not
    returned in ChatKitServer responses. If you serialize attachments from a custom
    direct-upload endpoint and want to omit this field, pass context={"exclude_metadata": True}.
    """

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info: SerializationInfo):
        data = serializer(self)
        if isinstance(data, dict) and (info.context or {}).get("exclude_metadata"):
            data.pop("metadata", None)
        return data


class FileAttachment(AttachmentBase):
    """Attachment representing a generic file."""

    type: Literal["file"] = "file"


class ImageAttachment(AttachmentBase):
    """Attachment representing an image resource."""

    type: Literal["image"] = "image"
    preview_url: AnyUrl


Attachment = Annotated[
    FileAttachment | ImageAttachment,
    Field(discriminator="type"),
]
"""Union of supported attachment types."""


### WORKFLOW TYPES


class Workflow(BaseModel):
    """Workflow attached to a thread with optional summary."""

    type: Literal["custom", "reasoning"]
    tasks: list[Task]
    summary: WorkflowSummary | None = None
    expanded: bool = False


class CustomSummary(BaseModel):
    """Custom summary for a workflow."""

    title: str
    icon: icons.IconName | None = None


class DurationSummary(BaseModel):
    """Summary providing total workflow duration."""

    duration: int
    """The duration of the workflow in seconds"""


WorkflowSummary = CustomSummary | DurationSummary
"""Summary variants available for workflows."""

### TASK TYPES


class BaseTask(BaseModel):
    """Base fields common to all workflow tasks."""

    status_indicator: Literal["none", "loading", "complete"] = "none"
    """Only used when rendering the task as part of a workflow. Indicates the status of the task."""


class CustomTask(BaseTask):
    """Workflow task displaying custom content."""

    type: Literal["custom"] = "custom"
    title: str | None = None
    icon: icons.IconName | None = None
    content: str | None = None


class SearchTask(BaseTask):
    """Workflow task representing a web search."""

    type: Literal["web_search"] = "web_search"
    title: str | None = None
    title_query: str | None = None
    queries: list[str] = Field(default_factory=list)
    sources: list[URLSource] = Field(default_factory=list)


class ThoughtTask(BaseTask):
    """Workflow task capturing assistant reasoning."""

    type: Literal["thought"] = "thought"
    title: str | None = None
    content: str


class FileTask(BaseTask):
    """Workflow task referencing file sources."""

    type: Literal["file"] = "file"
    title: str | None = None
    sources: list[FileSource] = Field(default_factory=list)


class ImageTask(BaseTask):
    """Workflow task rendering image content."""

    type: Literal["image"] = "image"
    title: str | None = None


Task = Annotated[
    CustomTask | SearchTask | ThoughtTask | FileTask | ImageTask,
    Field(discriminator="type"),
]
"""Union of workflow task variants."""


### SOURCE TYPES


class SourceBase(BaseModel):
    """Base class for sources displayed to users."""

    title: str
    description: str | None = None
    timestamp: str | None = None
    group: str | None = None


class FileSource(SourceBase):
    """Source metadata for file-based references."""

    type: Literal["file"] = "file"
    filename: str


class URLSource(SourceBase):
    """Source metadata for external URLs."""

    type: Literal["url"] = "url"
    url: str
    attribution: str | None = None


class EntitySource(SourceBase):
    """Source metadata for entity references."""

    type: Literal["entity"] = "entity"
    id: str
    icon: icons.IconName | None = None
    label: str | None = None
    """Optional label shown with the icon in the default entity hover header
    when no preview callback is provided.
    """
    inline_label: str | None = None
    """Optional label for the inline annotation view. When not provided, the icon is used instead."""
    interactive: bool = False
    """Per-entity toggle to wire client callbacks and render this entity as interactive."""
    data: dict[str, Any] = Field(default_factory=dict)
    """Additional data for the entity source that is passed to client entity callbacks."""


Source = Annotated[
    URLSource | FileSource | EntitySource,
    Field(discriminator="type"),
]
"""Union of supported source types."""


### MISC TYPES


FeedbackKind = Literal["positive", "negative"]
"""Literal type for feedback sentiment."""

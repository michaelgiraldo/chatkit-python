"""Stream events and updates delivered to ChatKit clients."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from . import errors, icons, models, widget_models


class ThreadCreatedEvent(BaseModel):
    """Event emitted when a thread is created."""

    type: Literal["thread.created"] = "thread.created"
    thread: models.Thread


class ThreadUpdatedEvent(BaseModel):
    """Event emitted when a thread is updated."""

    type: Literal["thread.updated"] = "thread.updated"
    thread: models.Thread


class ThreadItemAddedEvent(BaseModel):
    """Event emitted when a new item is added to a thread."""

    type: Literal["thread.item.added"] = "thread.item.added"
    item: models.ThreadItem


class ThreadItemUpdatedEvent(BaseModel):
    """Event describing an update to an existing thread item."""

    type: Literal["thread.item.updated"] = "thread.item.updated"
    item_id: str
    update: ThreadItemUpdate


class ThreadItemDoneEvent(BaseModel):
    """Event emitted when a thread item is marked complete."""

    type: Literal["thread.item.done"] = "thread.item.done"
    item: models.ThreadItem


class ThreadItemRemovedEvent(BaseModel):
    """Event emitted when a thread item is removed."""

    type: Literal["thread.item.removed"] = "thread.item.removed"
    item_id: str


class ThreadItemReplacedEvent(BaseModel):
    """Event emitted when a thread item is replaced."""

    type: Literal["thread.item.replaced"] = "thread.item.replaced"
    item: models.ThreadItem


class StreamOptions(BaseModel):
    """Settings that control runtime stream behavior."""

    allow_cancel: bool
    """Allow the client to request cancellation mid-stream."""


class StreamOptionsEvent(BaseModel):
    """Event emitted to set stream options at runtime."""

    type: Literal["stream_options"] = "stream_options"
    stream_options: StreamOptions


class ProgressUpdateEvent(BaseModel):
    """Event providing incremental progress from the assistant."""

    type: Literal["progress_update"] = "progress_update"
    icon: icons.IconName | None = None
    text: str


class ClientEffectEvent(BaseModel):
    """Event emitted to trigger a client side-effect."""

    type: Literal["client_effect"] = "client_effect"
    name: str
    data: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    """Event indicating an error occurred while processing a thread."""

    type: Literal["error"] = "error"
    code: errors.ErrorCode | Literal["custom"] = Field(default="custom")
    message: str | None = None
    allow_retry: bool = Field(default=False)


class NoticeEvent(BaseModel):
    """Event conveying a user-facing notice."""

    type: Literal["notice"] = "notice"
    level: Literal["info", "warning", "danger"]
    message: str
    """
    Supports markdown e.g. "You've reached your limit of 100 messages. [Upgrade](https://...) to a paid plan."
    """
    title: str | None = None


ThreadStreamEvent = Annotated[
    ThreadCreatedEvent
    | ThreadUpdatedEvent
    | ThreadItemDoneEvent
    | ThreadItemAddedEvent
    | ThreadItemUpdatedEvent
    | ThreadItemRemovedEvent
    | ThreadItemReplacedEvent
    | StreamOptionsEvent
    | ProgressUpdateEvent
    | ClientEffectEvent
    | ErrorEvent
    | NoticeEvent,
    Field(discriminator="type"),
]
"""Union of all streaming events emitted to clients."""

### THREAD ITEM UPDATE TYPES


class AssistantMessageContentPartAdded(BaseModel):
    """Event emitted when new assistant content is appended."""

    type: Literal["assistant_message.content_part.added"] = (
        "assistant_message.content_part.added"
    )
    content_index: int
    content: models.AssistantMessageContent


class AssistantMessageContentPartTextDelta(BaseModel):
    """Event carrying incremental assistant text output."""

    type: Literal["assistant_message.content_part.text_delta"] = (
        "assistant_message.content_part.text_delta"
    )
    content_index: int
    delta: str


class AssistantMessageContentPartAnnotationAdded(BaseModel):
    """Event announcing a new annotation on assistant content."""

    type: Literal["assistant_message.content_part.annotation_added"] = (
        "assistant_message.content_part.annotation_added"
    )
    content_index: int
    annotation_index: int
    annotation: models.Annotation


class AssistantMessageContentPartDone(BaseModel):
    """Event indicating an assistant content part is finalized."""

    type: Literal["assistant_message.content_part.done"] = (
        "assistant_message.content_part.done"
    )
    content_index: int
    content: models.AssistantMessageContent


class WidgetStreamingTextValueDelta(BaseModel):
    """Event streaming widget text deltas."""

    type: Literal["widget.streaming_text.value_delta"] = (
        "widget.streaming_text.value_delta"
    )
    component_id: str
    delta: str
    done: bool


class WidgetRootUpdated(BaseModel):
    """Event published when the widget root changes."""

    type: Literal["widget.root.updated"] = "widget.root.updated"
    widget: widget_models.WidgetRoot


class WidgetComponentUpdated(BaseModel):
    """Event emitted when a widget component updates."""

    type: Literal["widget.component.updated"] = "widget.component.updated"
    component_id: str
    component: widget_models.WidgetComponent


class WorkflowTaskAdded(BaseModel):
    """Event emitted when a workflow task is added."""

    type: Literal["workflow.task.added"] = "workflow.task.added"
    task_index: int
    task: models.Task


class WorkflowTaskUpdated(BaseModel):
    """Event emitted when a workflow task is updated."""

    type: Literal["workflow.task.updated"] = "workflow.task.updated"
    task_index: int
    task: models.Task


class GeneratedImageUpdated(BaseModel):
    """Event emitted when a generated image is updated."""

    type: Literal["generated_image.updated"] = "generated_image.updated"
    image: models.GeneratedImage
    progress: float | None = None


ThreadItemUpdate = (
    AssistantMessageContentPartAdded
    | AssistantMessageContentPartTextDelta
    | AssistantMessageContentPartAnnotationAdded
    | AssistantMessageContentPartDone
    | WidgetStreamingTextValueDelta
    | WidgetComponentUpdated
    | WidgetRootUpdated
    | WorkflowTaskAdded
    | WorkflowTaskUpdated
    | GeneratedImageUpdated
)
"""Union of possible updates applied to thread items."""


class SyncCustomActionResponse(BaseModel):
    """Single thread item update returned by a sync custom action."""

    updated_item: models.ThreadItem | None = None

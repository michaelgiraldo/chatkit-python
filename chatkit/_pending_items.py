"""Assemble independent copies of unfinished items for disconnect handling."""

from .events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    ThreadItemAddedEvent,
    ThreadItemUpdatedEvent,
    WorkflowTaskAdded,
    WorkflowTaskUpdated,
)
from .models import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItem,
    WorkflowItem,
)


def track_pending_item(
    pending: dict[str, ThreadItem], event: ThreadItemAddedEvent | ThreadItemUpdatedEvent
) -> None:
    if isinstance(event, ThreadItemAddedEvent):
        pending[event.item.id] = event.item.model_copy(deep=True)
        return

    item = pending.get(event.item_id)
    update = event.update
    if isinstance(item, AssistantMessageItem) and isinstance(
        update,
        (
            AssistantMessageContentPartAdded,
            AssistantMessageContentPartDone,
            AssistantMessageContentPartTextDelta,
            AssistantMessageContentPartAnnotationAdded,
        ),
    ):
        # A delta can arrive before its content part has been added.
        while len(item.content) <= update.content_index:
            item.content.append(AssistantMessageContent(text=""))

        match update:
            case AssistantMessageContentPartAdded() | AssistantMessageContentPartDone():
                item.content[update.content_index] = update.content.model_copy(
                    deep=True
                )
            case AssistantMessageContentPartTextDelta():
                item.content[update.content_index].text += update.delta
            case AssistantMessageContentPartAnnotationAdded():
                item.content[update.content_index].annotations.insert(
                    update.annotation_index, update.annotation.model_copy(deep=True)
                )
    elif isinstance(item, WorkflowItem):
        match update:
            case WorkflowTaskUpdated():
                item.workflow.tasks[update.task_index] = update.task.model_copy(
                    deep=True
                )
            case WorkflowTaskAdded():
                item.workflow.tasks.append(update.task.model_copy(deep=True))

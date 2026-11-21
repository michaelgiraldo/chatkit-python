"""Project stored threads and responder events into browser-visible ChatKit output."""

from .events import (
    SyncCustomActionResponse,
    ThreadCreatedEvent,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemRemovedEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    ThreadUpdatedEvent,
)
from .models import Page, Thread, ThreadItem, ThreadMetadata
from .store import HIDDEN_ITEM_TYPES


def browser_thread(
    thread: ThreadMetadata,
    *,
    items: Page[ThreadItem] | None = None,
    hidden_item_ids: set[str] | None = None,
) -> Thread:
    """Copy public thread fields and retain the store's pagination cursor."""
    if items is None and isinstance(thread, Thread):
        items = thread.items
    if items is not None:
        visible = []
        for item in items.data:
            if item.type not in HIDDEN_ITEM_TYPES:
                visible.append(item)
            elif hidden_item_ids is not None:
                hidden_item_ids.add(item.id)
        items = items.model_copy(update={"data": visible})
    return Thread(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        items=items if items is not None else Page(),
        status=thread.status,
        allowed_image_domains=thread.allowed_image_domains,
    )


def browser_widget_response(
    response: SyncCustomActionResponse,
) -> SyncCustomActionResponse:
    if (
        response.updated_item is not None
        and response.updated_item.type in HIDDEN_ITEM_TYPES
    ):
        return SyncCustomActionResponse()
    return response


def browser_event(
    event: ThreadStreamEvent, hidden_item_ids: set[str]
) -> ThreadStreamEvent | None:
    """Track hidden IDs within a stream so later ID-only updates stay private."""
    match event:
        case ThreadCreatedEvent(thread=thread) | ThreadUpdatedEvent(thread=thread):
            return event.model_copy(
                update={
                    "thread": browser_thread(thread, hidden_item_ids=hidden_item_ids)
                }
            )
        case (
            ThreadItemAddedEvent(item=item)
            | ThreadItemDoneEvent(item=item)
            | ThreadItemReplacedEvent(item=item)
        ) if item.type in HIDDEN_ITEM_TYPES:
            hidden_item_ids.add(item.id)
            return None
        case (
            ThreadItemUpdatedEvent(item_id=item_id)
            | ThreadItemRemovedEvent(item_id=item_id)
        ) if item_id in hidden_item_ids:
            return None
    return event

"""Convert Responses message and image events into ChatKit transcript items."""

from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from openai.types.responses import ResponseOutputMessage, ResponseStreamEvent
from openai.types.responses.response_output_message import Content
from openai.types.responses.response_output_text import (
    Annotation as ResponsesAnnotation,
)
from pydantic import BaseModel, TypeAdapter

from .events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    GeneratedImageUpdated,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
)
from .models import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageItem,
    AssistantMessagePhase,
    GeneratedImage,
    GeneratedImageItem,
)

if TYPE_CHECKING:
    from .agents import AgentContext, ResponseStreamConverter


_CHAT_COMPLETIONS_PLACEHOLDER_ID = "__fake_id__"
_ANNOTATION_ADAPTER = TypeAdapter[ResponsesAnnotation](ResponsesAnnotation)


async def _convert_content(
    content: Content, converter: ResponseStreamConverter
) -> AssistantMessageContent:
    if content.type == "output_text":
        annotations = [
            await _convert_annotation(annotation, converter)
            for annotation in content.annotations
        ]
        annotations = [a for a in annotations if a is not None]
        return AssistantMessageContent(
            text=content.text,
            annotations=annotations,
        )
    else:
        return AssistantMessageContent(
            text=content.refusal,
            annotations=[],
        )


async def _convert_annotation(
    raw_annotation: object, converter: ResponseStreamConverter
) -> Annotation | None:
    if raw_annotation is None:
        return None

    # Stream events can use different model classes than completed response content.
    if isinstance(raw_annotation, BaseModel):
        raw_annotation = raw_annotation.model_dump()

    annotation = _ANNOTATION_ADAPTER.validate_python(raw_annotation)

    if annotation.type == "file_citation":
        return await converter.file_citation_to_annotation(annotation)

    if annotation.type == "url_citation":
        return await converter.url_citation_to_annotation(annotation)

    if annotation.type == "container_file_citation":
        return await converter.container_file_citation_to_annotation(annotation)

    return None


class _StreamedOutput:
    """Track provider item identity, citations and partial images for one streaming run."""

    def __init__(
        self,
        context: AgentContext,
        converter: ResponseStreamConverter,
        produced_items: set[str],
    ) -> None:
        self._context = context
        self._converter = converter
        self._produced_items = produced_items
        self._synthetic_ids: dict[int, str] = {}
        self._message_phases: dict[str, AssistantMessagePhase] = {}
        self._annotations: Counter[tuple[str, int]] = Counter()
        self._images: dict[str, GeneratedImageItem] = {}

    def _message_id(self, item_id: str, output_index: int) -> str:
        # Chat Completions streams a placeholder. Keep one store ID until this item completes.
        if item_id != _CHAT_COMPLETIONS_PLACEHOLDER_ID:
            return item_id
        if output_index not in self._synthetic_ids:
            self._synthetic_ids[output_index] = self._context.generate_id("message")
        return self._synthetic_ids[output_index]

    async def _message(
        self, item: ResponseOutputMessage, output_index: int, *, done: bool
    ) -> ThreadItemAddedEvent | ThreadItemDoneEvent:
        item_id = self._message_id(item.id, output_index)
        phase = item.phase
        if done:
            initial_phase = self._message_phases.pop(item_id, None)
            phase = phase or initial_phase
            if item.id == _CHAT_COMPLETIONS_PLACEHOLDER_ID:
                del self._synthetic_ids[output_index]
        elif phase is not None:
            self._message_phases[item_id] = phase
        self._produced_items.add(item_id)
        message = AssistantMessageItem(
            id=item_id,
            thread_id=self._context.thread.id,
            phase=phase,
            content=[
                await _convert_content(part, self._converter) for part in item.content
            ],
            created_at=datetime.now(UTC),
        )
        return (
            ThreadItemDoneEvent(item=message)
            if done
            else ThreadItemAddedEvent(item=message)
        )

    async def handle(
        self, event: ResponseStreamEvent
    ) -> ThreadItemAddedEvent | ThreadItemUpdatedEvent | ThreadItemDoneEvent | None:
        if event.type == "response.output_item.added":
            if event.item.type == "message":
                return await self._message(event.item, event.output_index, done=False)
            if event.item.type == "image_generation_call":
                image = GeneratedImageItem(
                    id=self._context.generate_id("message"),
                    thread_id=self._context.thread.id,
                    created_at=datetime.now(UTC),
                    image=None,
                )
                self._images[event.item.id] = image
                self._produced_items.add(image.id)
                return ThreadItemAddedEvent(item=image)

        elif event.type == "response.output_item.done":
            if event.item.type == "message":
                return await self._message(event.item, event.output_index, done=True)
            if event.item.type == "image_generation_call":
                image = self._images.pop(event.item.id, None)
                if image is not None and event.item.result:
                    url = await self._converter.base64_image_to_url(
                        image_id=event.item.id, base64_image=event.item.result
                    )
                    return ThreadItemDoneEvent(
                        item=image.model_copy(
                            update={"image": GeneratedImage(id=event.item.id, url=url)}
                        )
                    )

        elif event.type == "response.image_generation_call.partial_image":
            if (image := self._images.get(event.item_id)) is not None:
                url = await self._converter.base64_image_to_url(
                    image_id=event.item_id,
                    base64_image=event.partial_image_b64,
                    partial_image_index=event.partial_image_index,
                )
                progress = self._converter.partial_image_index_to_progress(
                    event.partial_image_index
                )
                return ThreadItemUpdatedEvent(
                    item_id=image.id,
                    update=GeneratedImageUpdated(
                        image=GeneratedImage(id=event.item_id, url=url),
                        progress=progress,
                    ),
                )

        elif event.type == "response.content_part.added":
            if event.part.type != "reasoning_text":
                content = await _convert_content(event.part, self._converter)
                return ThreadItemUpdatedEvent(
                    item_id=self._message_id(event.item_id, event.output_index),
                    update=AssistantMessageContentPartAdded(
                        content_index=event.content_index, content=content
                    ),
                )

        elif event.type == "response.output_text.delta":
            return ThreadItemUpdatedEvent(
                item_id=self._message_id(event.item_id, event.output_index),
                update=AssistantMessageContentPartTextDelta(
                    content_index=event.content_index, delta=event.delta
                ),
            )

        elif event.type == "response.output_text.done":
            return ThreadItemUpdatedEvent(
                item_id=self._message_id(event.item_id, event.output_index),
                update=AssistantMessageContentPartDone(
                    content_index=event.content_index,
                    content=AssistantMessageContent(text=event.text, annotations=[]),
                ),
            )

        elif event.type == "response.output_text.annotation.added":
            annotation = await _convert_annotation(event.annotation, self._converter)
            if annotation is not None:
                item_id = self._message_id(event.item_id, event.output_index)
                # ChatKit indexes only the annotations that were sent to the browser.
                key = (item_id, event.content_index)
                index = self._annotations[key]
                self._annotations[key] += 1
                return ThreadItemUpdatedEvent(
                    item_id=item_id,
                    update=AssistantMessageContentPartAnnotationAdded(
                        content_index=event.content_index,
                        annotation_index=index,
                        annotation=annotation,
                    ),
                )
        return None

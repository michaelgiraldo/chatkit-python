from unittest.mock import AsyncMock, Mock

import pytest
from agents import (
    RawResponsesStreamEvent,
)
from openai.types.responses import (
    ResponseFileSearchToolCall,
    ResponseImageGenCallPartialImageEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputTextAnnotationAddedEvent,
)
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from openai.types.responses.response_file_search_tool_call import Result
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_output_text import (
    AnnotationContainerFileCitation as ResponsesAnnotationContainerFileCitation,
)
from openai.types.responses.response_output_text import (
    AnnotationFileCitation as ResponsesAnnotationFileCitation,
)
from openai.types.responses.response_output_text import (
    AnnotationFilePath as ResponsesAnnotationFilePath,
)
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponsesAnnotationURLCitation,
)
from openai.types.responses.response_output_text import (
    ResponseOutputText,
)
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_text_done_event import ResponseTextDoneEvent

from chatkit.agents import (
    AgentContext,
    ResponseStreamConverter,
    stream_agent_response,
)
from chatkit.events import (
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartAnnotationAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    GeneratedImageUpdated,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemUpdatedEvent,
)
from chatkit.models import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageItem,
    FileSource,
    GeneratedImage,
    GeneratedImageItem,
    Page,
    URLSource,
)
from tests.helpers.agent_stream import RunResult, all_events, make_thread_and_store

thread, mock_store = make_thread_and_store()


async def test_chat_completions_allocates_a_new_id_when_an_output_index_is_reused():
    store = Mock(
        generate_item_id=Mock(side_effect=["message_1", "message_2"]),
        load_thread_items=AsyncMock(return_value=Page()),
    )
    context = AgentContext(
        thread=thread, store=store, request_context=None, resume_from=None
    )
    result = RunResult()
    placeholder = ResponseOutputMessage(
        id="__fake_id__",
        content=[],
        role="assistant",
        status="in_progress",
        type="message",
    )
    for sequence_number in (0, 2):
        result.add_event(
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    item=placeholder,
                    output_index=0,
                    sequence_number=sequence_number,
                ),
            )
        )
        result.add_event(
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    item=placeholder.model_copy(update={"status": "completed"}),
                    output_index=0,
                    sequence_number=sequence_number + 1,
                ),
            )
        )
    result.done()

    events = await all_events(stream_agent_response(context, result))
    assert [
        (event.type, event.item.id)
        for event in events
        if isinstance(event, (ThreadItemAddedEvent, ThreadItemDoneEvent))
    ] == [
        ("thread.item.added", "message_1"),
        ("thread.item.done", "message_1"),
        ("thread.item.added", "message_2"),
        ("thread.item.done", "message_2"),
    ]


@pytest.mark.parametrize(
    "raw_event,expected_event",
    [
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    delta="Hello, world!",
                    content_index=0,
                    item_id="123",
                    logprobs=[],
                    output_index=0,
                    sequence_number=0,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartTextDelta(
                    content_index=0,
                    delta="Hello, world!",
                ),
            ),
        ),
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    part=ResponseOutputText(
                        type="output_text",
                        text="New content",
                        annotations=[],
                    ),
                    content_index=1,
                    item_id="123",
                    output_index=0,
                    sequence_number=1,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartAdded(
                    content_index=1,
                    content=AssistantMessageContent(text="New content", annotations=[]),
                ),
            ),
        ),
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=ResponseTextDoneEvent(
                    type="response.output_text.done",
                    text="Final text",
                    content_index=0,
                    item_id="123",
                    logprobs=[],
                    output_index=0,
                    sequence_number=2,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartDone(
                    content_index=0,
                    content=AssistantMessageContent(
                        text="Final text",
                        annotations=[],
                    ),
                ),
            ),
        ),
        (
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=Mock(
                    type="response.output_text.annotation.added",
                    annotation=ResponsesAnnotationFileCitation(
                        type="file_citation",
                        file_id="file_123",
                        filename="file.txt",
                        index=5,
                    ),
                    content_index=0,
                    item_id="123",
                    annotation_index=0,
                    output_index=0,
                    sequence_number=3,
                ),
            ),
            ThreadItemUpdatedEvent(
                item_id="123",
                update=AssistantMessageContentPartAnnotationAdded(
                    content_index=0,
                    annotation_index=0,
                    annotation=Annotation(
                        source=FileSource(filename="file.txt", title="file.txt"),
                        index=5,
                    ),
                ),
            ),
        ),
    ],
)
async def test_event_mapping(raw_event, expected_event):
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

    result.add_event(raw_event)
    result.done()

    events = await all_events(stream_agent_response(context, result))
    if expected_event:
        assert events == [expected_event]
    else:
        assert events == []


async def test_stream_agent_response_emits_annotation_added_events():
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()
    item_id = "item_123"

    def add_annotation_event(annotation, sequence_number):
        result.add_event(
            RawResponsesStreamEvent(
                type="raw_response_event",
                data=Mock(
                    type="response.output_text.annotation.added",
                    annotation=annotation,
                    content_index=0,
                    item_id=item_id,
                    annotation_index=sequence_number,
                    output_index=0,
                    sequence_number=sequence_number,
                ),
            )
        )

    add_annotation_event(
        ResponsesAnnotationFileCitation(
            type="file_citation",
            file_id="file_invalid",
            filename="",
            index=0,
        ),
        sequence_number=0,
    )
    add_annotation_event(
        ResponsesAnnotationContainerFileCitation(
            type="container_file_citation",
            container_id="container_1",
            file_id="file_123",
            filename="container.txt",
            start_index=0,
            end_index=3,
        ),
        sequence_number=1,
    )
    add_annotation_event(
        ResponsesAnnotationURLCitation(
            type="url_citation",
            url="https://example.com",
            title="Example",
            start_index=1,
            end_index=5,
        ),
        sequence_number=2,
    )
    result.done()

    events = await all_events(stream_agent_response(context, result))
    assert events == [
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=0,
                annotation=Annotation(
                    source=FileSource(filename="container.txt", title="container.txt"),
                    index=3,
                ),
            ),
        ),
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=1,
                annotation=Annotation(
                    source=URLSource(
                        url="https://example.com",
                        title="Example",
                    ),
                    index=5,
                ),
            ),
        ),
    ]


async def test_stream_agent_response_annotation_added_normalizes_annotations():
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()
    item_id = "item_123"

    def add_annotation_event(
        annotation: dict[str, object] | None, sequence_number: int
    ):
        event = ResponseOutputTextAnnotationAddedEvent.model_validate({
            "type": "response.output_text.annotation.added",
            "annotation": annotation,
            "content_index": 0,
            "item_id": item_id,
            "annotation_index": sequence_number,
            "output_index": 0,
            "sequence_number": sequence_number,
        })
        result.add_event(RawResponsesStreamEvent(type="raw_response_event", data=event))

    # Invalid file citation is dropped (empty filename)
    add_annotation_event(
        {
            "type": "file_citation",
            "file_id": "file_invalid",
            "filename": "",
            "index": 0,
        },
        sequence_number=0,
    )

    add_annotation_event(None, sequence_number=1)

    # The next citations still use indices 0 and 1 after the dropped events.
    add_annotation_event(
        {
            "type": "container_file_citation",
            "container_id": "container_1",
            "file_id": "file_123",
            "filename": "container.txt",
            "start_index": 0,
            "end_index": 3,
        },
        sequence_number=2,
    )
    add_annotation_event(
        {
            "type": "url_citation",
            "url": "https://example.com",
            "title": "Example",
            "start_index": 1,
            "end_index": 5,
        },
        sequence_number=3,
    )

    result.done()

    events = await all_events(stream_agent_response(context, result))
    assert events == [
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=0,
                annotation=Annotation(
                    source=FileSource(filename="container.txt", title="container.txt"),
                    index=3,
                ),
            ),
        ),
        ThreadItemUpdatedEvent(
            item_id=item_id,
            update=AssistantMessageContentPartAnnotationAdded(
                content_index=0,
                annotation_index=1,
                annotation=Annotation(
                    source=URLSource(url="https://example.com", title="Example"),
                    index=5,
                ),
            ),
        ),
    ]


async def test_custom_annotation_conversion_used_by_stream_agent_response():
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

    class CustomResponseStreamConverter(ResponseStreamConverter):
        def __init__(self):
            super().__init__()
            self.calls: list[str] = []

        async def file_citation_to_annotation(self, file_citation):
            self.calls.append("file_citation")
            return Annotation(
                source=FileSource(
                    filename="report.pdf",
                    title="Usage Report",
                    description="Usage report for the month of January",
                ),
                index=111,
            )

        async def url_citation_to_annotation(self, url_citation):
            self.calls.append("url_citation")
            return Annotation(
                source=URLSource(url="https://custom.example/url", title="Custom"),
                index=222,
            )

    converter = CustomResponseStreamConverter()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_text.annotation.added",
                annotation=ResponsesAnnotationFileCitation(
                    type="file_citation",
                    file_id="file_123",
                    filename="file.txt",
                    index=0,
                ),
                content_index=0,
                item_id="m_1",
                annotation_index=0,
                output_index=0,
                sequence_number=0,
            ),
        )
    )

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseOutputMessage(
                    id="m_1",
                    content=[
                        ResponseOutputText(
                            annotations=[
                                ResponsesAnnotationURLCitation(
                                    type="url_citation",
                                    url="https://example.com",
                                    title="Example",
                                    start_index=0,
                                    end_index=7,
                                )
                            ],
                            text="Hello!",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.done()

    events = await all_events(
        stream_agent_response(context, result, converter=converter)
    )
    assert len(events) == 2

    assert isinstance(events[0], ThreadItemUpdatedEvent)
    assert events[0].item_id == "m_1"
    assert events[0].update == AssistantMessageContentPartAnnotationAdded(
        content_index=0,
        annotation_index=0,
        annotation=Annotation(
            source=FileSource(
                filename="report.pdf",
                title="Usage Report",
                description="Usage report for the month of January",
            ),
            index=111,
        ),
    )

    assert isinstance(events[1], ThreadItemDoneEvent)
    assert isinstance(events[1].item, AssistantMessageItem)
    assert events[1].item.id == "m_1"
    assert events[1].item.content == [
        AssistantMessageContent(
            text="Hello!",
            annotations=[
                Annotation(
                    source=URLSource(url="https://custom.example/url", title="Custom"),
                    index=222,
                )
            ],
        )
    ]
    assert converter.calls == ["file_citation", "url_citation"]


async def test_stream_agent_response_assistant_message_content_types():
    AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseFileSearchToolCall(
                    id="fs_0",
                    queries=["Hello, world!"],
                    status="completed",
                    type="file_search_call",
                    results=[
                        Result(
                            file_id="f_123",
                            filename="test.txt",
                            text="Hello, world!",
                            score=1.0,
                        ),
                        Result(
                            file_id="f_123",
                            filename="test.txt",
                            text="Hello, friends!",
                            score=0.5,
                        ),
                    ],
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=ResponseOutputMessage(
                    id="1",
                    content=[
                        ResponseOutputText(
                            annotations=[
                                ResponsesAnnotationFileCitation(
                                    type="file_citation",
                                    file_id="f_123",
                                    index=0,
                                    filename="test.txt",
                                ),
                                ResponsesAnnotationContainerFileCitation(
                                    type="container_file_citation",
                                    container_id="container_1",
                                    file_id="f_456",
                                    filename="container.txt",
                                    start_index=0,
                                    end_index=3,
                                ),
                                ResponsesAnnotationURLCitation(
                                    type="url_citation",
                                    url="https://www.google.com",
                                    title="Google",
                                    start_index=0,
                                    end_index=10,
                                ),
                                ResponsesAnnotationFilePath(
                                    type="file_path",
                                    file_id="123",
                                    index=0,
                                ),
                            ],
                            text="Hello, world!",
                            type="output_text",
                        ),
                        ResponseOutputText(
                            annotations=[],
                            text="Can't do that",
                            type="output_text",
                        ),
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                output_index=0,
                sequence_number=0,
            ),
        )
    )

    result.done()

    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    events = await all_events(stream_agent_response(context, result))
    assert len(events) == 1
    assert isinstance(events[0], ThreadItemDoneEvent)
    message = events[0].item
    assert isinstance(message, AssistantMessageItem)
    assert message.content == [
        AssistantMessageContent(
            annotations=[
                Annotation(
                    source=FileSource(
                        filename="test.txt",
                        title="test.txt",
                    ),
                    index=0,
                ),
                Annotation(
                    source=FileSource(
                        filename="container.txt",
                        title="container.txt",
                    ),
                    index=3,
                ),
                Annotation(
                    source=URLSource(
                        url="https://www.google.com",
                        title="Google",
                    ),
                    index=10,
                ),
            ],
            text="Hello, world!",
        ),
        AssistantMessageContent(text="Can't do that", annotations=[]),
    ]
    assert message.id == "1"


async def test_stream_agent_response_image_generation_events():
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.added",
                item=Mock(type="image_generation_call", id="img_call_1"),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.done",
                item=Mock(
                    type="image_generation_call", id="img_call_1", result="dGVzdA=="
                ),
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.done()

    stream = stream_agent_response(context, result)
    event1 = await stream.__anext__()
    assert isinstance(event1, ThreadItemAddedEvent)
    assert isinstance(event1.item, GeneratedImageItem)
    assert event1.item.type == "generated_image"
    assert event1.item.id == "message_id"
    assert event1.item.image is None

    event2 = await stream.__anext__()
    assert isinstance(event2, ThreadItemDoneEvent)
    assert isinstance(event2.item, GeneratedImageItem)
    assert event2.item.id == event1.item.id
    assert event2.item.image == GeneratedImage(
        id="img_call_1", url="data:image/png;base64,dGVzdA=="
    )

    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_stream_agent_response_image_generation_events_with_custom_converter():
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.added",
                item=Mock(type="image_generation_call", id="img_call_1"),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.done",
                item=Mock(
                    type="image_generation_call", id="img_call_1", result="dGVzdA=="
                ),
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.done()

    class CustomResponseStreamConverter(ResponseStreamConverter):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[str, str, int | None]] = []

        async def base64_image_to_url(
            self,
            image_id: str,
            base64_image: str,
            partial_image_index: int | None = None,
        ) -> str:
            self.calls.append((image_id, base64_image, partial_image_index))
            return f"https://example.com/{image_id}"

    converter = CustomResponseStreamConverter()
    stream = stream_agent_response(context, result, converter=converter)
    event1 = await stream.__anext__()
    assert isinstance(event1, ThreadItemAddedEvent)
    assert isinstance(event1.item, GeneratedImageItem)
    assert event1.item.image is None

    event2 = await stream.__anext__()
    assert isinstance(event2, ThreadItemDoneEvent)
    assert isinstance(event2.item, GeneratedImageItem)
    assert converter.calls == [("img_call_1", "dGVzdA==", None)]
    assert event2.item.image == GeneratedImage(
        id="img_call_1", url="https://example.com/img_call_1"
    )
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


async def test_stream_agent_response_image_generation_partial_progress():
    context = AgentContext(
        thread=thread, store=mock_store, request_context=None, resume_from=None
    )
    result = RunResult()

    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.added",
                item=Mock(type="image_generation_call", id="img_call_1"),
                output_index=0,
                sequence_number=0,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.image_generation_call.partial_image",
                partial_image_b64="dGVzdA==",
                partial_image_index=1,
                item_id="img_call_1",
                output_index=0,
                sequence_number=1,
            ),
        )
    )
    result.add_event(
        RawResponsesStreamEvent(
            type="raw_response_event",
            data=Mock(
                type="response.output_item.done",
                item=Mock(
                    type="image_generation_call", id="img_call_1", result="dGVzdA=="
                ),
                output_index=0,
                sequence_number=2,
            ),
        )
    )
    result.done()

    converter = ResponseStreamConverter(partial_images=3)
    events = await all_events(
        stream_agent_response(context, result, converter=converter)
    )

    assert len(events) == 3
    added_event, partial_event, done_event = events

    assert isinstance(added_event, ThreadItemAddedEvent)
    assert isinstance(added_event.item, GeneratedImageItem)

    assert isinstance(partial_event, ThreadItemUpdatedEvent)
    assert isinstance(partial_event.update, GeneratedImageUpdated)
    assert partial_event.update.progress == pytest.approx(1 / 3)
    assert partial_event.update.image == GeneratedImage(
        id="img_call_1", url="data:image/png;base64,dGVzdA=="
    )

    assert isinstance(done_event, ThreadItemDoneEvent)
    assert isinstance(done_event.item, GeneratedImageItem)
    assert done_event.item.image == GeneratedImage(
        id="img_call_1", url="data:image/png;base64,dGVzdA=="
    )


async def test_interleaved_image_generation_keeps_each_preview_and_result():
    store = Mock(
        generate_item_id=Mock(side_effect=["image_1", "image_2"]),
        load_thread_items=AsyncMock(return_value=Page()),
    )
    context = AgentContext(
        thread=thread, store=store, request_context=None, resume_from=None
    )
    result = RunResult()
    first = ImageGenerationCall(
        id="img_call_1", status="in_progress", type="image_generation_call"
    )
    second = ImageGenerationCall(
        id="img_call_2", status="in_progress", type="image_generation_call"
    )

    for event in [
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=first,
            output_index=0,
            sequence_number=0,
        ),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            item=second,
            output_index=1,
            sequence_number=1,
        ),
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=first.id,
            output_index=0,
            partial_image_index=1,
            partial_image_b64="MQ==",
            sequence_number=2,
        ),
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=second.id,
            output_index=1,
            partial_image_index=1,
            partial_image_b64="Mg==",
            sequence_number=3,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=second.model_copy(update={"status": "completed", "result": "Mw=="}),
            output_index=1,
            sequence_number=4,
        ),
        # A late preview for the completed call must not update the pending call.
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=second.id,
            output_index=1,
            partial_image_index=2,
            partial_image_b64="MA==",
            sequence_number=5,
        ),
        ResponseImageGenCallPartialImageEvent(
            type="response.image_generation_call.partial_image",
            item_id=first.id,
            output_index=0,
            partial_image_index=2,
            partial_image_b64="NA==",
            sequence_number=6,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=first.model_copy(update={"status": "completed", "result": "NQ=="}),
            output_index=0,
            sequence_number=7,
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=ResponseOutputMessage(
                id="assistant_message",
                type="message",
                role="assistant",
                status="completed",
                content=[],
            ),
            output_index=2,
            sequence_number=8,
        ),
    ]:
        result.add_event(RawResponsesStreamEvent(type="raw_response_event", data=event))
    result.done()

    events = await all_events(
        stream_agent_response(
            context, result, converter=ResponseStreamConverter(partial_images=3)
        )
    )
    assert [getattr(event, "type", None) for event in events] == [
        "thread.item.added",
        "thread.item.added",
        "thread.item.updated",
        "thread.item.updated",
        "thread.item.done",
        "thread.item.updated",
        "thread.item.done",
        "thread.item.done",
    ]
    for event, item_id in zip(events[:2], ("image_1", "image_2"), strict=True):
        assert isinstance(event, ThreadItemAddedEvent)
        assert isinstance(event.item, GeneratedImageItem)
        assert event.item.id == item_id
        assert event.item.image is None

    for index, item_id, provider_id, image_data, progress in (
        (2, "image_1", first.id, "MQ==", 1 / 3),
        (3, "image_2", second.id, "Mg==", 1 / 3),
        (5, "image_1", first.id, "NA==", 2 / 3),
    ):
        event = events[index]
        assert isinstance(event, ThreadItemUpdatedEvent)
        assert event.item_id == item_id
        assert isinstance(event.update, GeneratedImageUpdated)
        assert event.update.image == GeneratedImage(
            id=provider_id, url=f"data:image/png;base64,{image_data}"
        )
        assert event.update.progress == pytest.approx(progress)

    for index, item_id, provider_id, image_data in (
        (4, "image_2", second.id, "Mw=="),
        (6, "image_1", first.id, "NQ=="),
    ):
        event = events[index]
        assert isinstance(event, ThreadItemDoneEvent)
        assert isinstance(event.item, GeneratedImageItem)
        assert event.item.id == item_id
        assert event.item.image == GeneratedImage(
            id=provider_id, url=f"data:image/png;base64,{image_data}"
        )

    message = events[-1]
    assert isinstance(message, ThreadItemDoneEvent)
    assert isinstance(message.item, AssistantMessageItem)
    assert message.item.id == "assistant_message"

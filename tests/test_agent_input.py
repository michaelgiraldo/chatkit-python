from datetime import datetime
from typing import cast

import pytest
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputContentParam,
    ResponseInputTextParam,
)
from openai.types.responses.response_input_item_param import Message
from pydantic import JsonValue

from chatkit.agent_input import (
    ThreadItemConverter,
    client_tool_continuation_input,
    simple_to_agent_input,
)
from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    Attachment,
    ClientToolCallItem,
    ClientToolLocalContinuation,
    CustomTask,
    GeneratedImage,
    GeneratedImageItem,
    HiddenContextItem,
    InferenceOptions,
    StructuredInputAnswer,
    StructuredInputFreeform,
    StructuredInputItem,
    StructuredInputMultipleChoice,
    StructuredInputMultipleChoiceOption,
    TaskItem,
    UserMessageItem,
    UserMessageTagContent,
    UserMessageTextContent,
    WidgetItem,
)
from chatkit.widgets import Card, Text

_THREAD_ID = "123"


async def test_input_item_converter_quotes_last_user_message():
    items = [
        UserMessageItem(
            id="123",
            content=[UserMessageTextContent(text="Hello!")],
            attachments=[],
            inference_options=InferenceOptions(),
            thread_id=_THREAD_ID,
            quoted_text="Hi!",
            created_at=datetime.now(),
        ),
        UserMessageItem(
            id="123",
            content=[UserMessageTextContent(text="I'm well, thank you!")],
            attachments=[],
            inference_options=InferenceOptions(),
            thread_id=_THREAD_ID,
            quoted_text="How are you doing?",
            created_at=datetime.now(),
        ),
    ]

    async def throw_exception(
        _: Attachment,
    ) -> ResponseInputContentParam:
        raise Exception("Not implemented")

    input_items = await simple_to_agent_input(items)
    assert len(input_items) == 3
    assert input_items[0] == {
        "content": [
            {
                "text": "Hello!",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }
    assert input_items[1] == {
        "content": [
            {
                "text": "I'm well, thank you!",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }
    assert input_items[2] == {
        "content": [
            {
                "text": "The user is referring to this in particular: \nHow are you doing?",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }


async def test_input_item_converter_to_input_items_mixed():
    items = [
        UserMessageItem(
            id="123",
            content=[UserMessageTextContent(text="Hello!")],
            attachments=[],
            inference_options=InferenceOptions(),
            thread_id=_THREAD_ID,
            quoted_text="Hi!",
            created_at=datetime.now(),
        ),
        UserMessageItem(
            id="123",
            content=[UserMessageTextContent(text="I'm well, thank you!")],
            attachments=[],
            inference_options=InferenceOptions(),
            thread_id=_THREAD_ID,
            quoted_text="How are you doing?",
            created_at=datetime.now(),
        ),
        AssistantMessageItem(
            id="123",
            content=[
                AssistantMessageContent(text="How are you doing?"),
                AssistantMessageContent(text="Can't do that"),
            ],
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
        ),
        WidgetItem(
            id="wd_123",
            widget=Card(children=[Text(value="Hello, world!")]).build(),
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
        ),
    ]

    input_items = await simple_to_agent_input(items)
    assert len(input_items) == 4
    assert input_items[0] == {
        "content": [
            {
                "text": "Hello!",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }
    assert input_items[1] == {
        "content": [
            {
                "text": "I'm well, thank you!",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }
    assert input_items[2] == {
        "content": [
            {
                "annotations": [],
                "text": "How are you doing?",
                "logprobs": None,
                "type": "output_text",
            },
            {
                "annotations": [],
                "text": "Can't do that",
                "logprobs": None,
                "type": "output_text",
            },
        ],
        "type": "message",
        "role": "assistant",
    }
    assert "type" in input_items[3]
    widget_item = cast(EasyInputMessageParam, input_items[3])
    assert widget_item.get("type") == "message"
    assert widget_item.get("role") == "user"
    text = widget_item.get("content")[0]["text"]  # type: ignore
    assert (
        "The following graphical UI widget (id: wd_123) was displayed to the user"
        in text
    )
    assert "Hello, world!" in text
    assert "created_at" not in text


async def test_input_item_converter_user_input_with_tags():
    class MyThreadItemConverter(ThreadItemConverter):
        async def tag_to_message_content(self, tag):
            return ResponseInputTextParam(
                type="input_text", text=tag.text + " " + tag.data["key"]
            )

    items = [
        UserMessageItem(
            id="123",
            content=[
                UserMessageTagContent(
                    text="Hello!", type="input_tag", id="hello", data={"key": "value"}
                )
            ],
            attachments=[],
            inference_options=InferenceOptions(),
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
        )
    ]
    items = await MyThreadItemConverter().to_agent_input(items)

    assert len(items) == 2
    assert items[0] == {
        "content": [
            {
                "text": "@Hello!",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }
    assert items[1] == {
        "content": [
            {
                "text": "# User-provided context for @-mentions\n- When referencing resolved entities, use their canonical names **without** '@'.\n"
                + "- The '@' form appears only in user text and should not be echoed.",
                "type": "input_text",
            },
            {
                "text": "Hello! value",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }


async def test_input_item_converter_user_input_with_tags_throws_by_default():
    items = [
        UserMessageItem(
            id="123",
            content=[
                UserMessageTagContent(
                    text="Hello!", type="input_tag", id="hello", data={}
                )
            ],
            attachments=[],
            inference_options=InferenceOptions(),
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
        )
    ]

    with pytest.raises(NotImplementedError):
        await simple_to_agent_input(items)


async def test_input_item_converter_generated_image_item():
    items = [
        GeneratedImageItem(
            id="img_item_1",
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
            image=GeneratedImage(id="img_1", url="https://example.com/img.png"),
        )
    ]

    input_items = await simple_to_agent_input(items)
    assert len(input_items) == 1

    message = cast(dict, input_items[0])
    assert message.get("type") == "message"
    assert message.get("role") == "user"

    content = cast(list, message.get("content"))
    assert content[0].get("type") == "input_text"
    assert content[0].get("text") == "The following image was generated by the agent."
    assert content[1].get("type") == "input_image"
    assert content[1].get("file_id") is None
    assert content[1].get("image_url") == "https://example.com/img.png"
    assert content[1].get("detail") == "auto"


async def test_input_item_converter_generated_image_item_without_image():
    items = [
        GeneratedImageItem(
            id="img_item_1",
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
        )
    ]

    input_items = await simple_to_agent_input(items)
    assert input_items == []


async def test_input_item_converter_for_hidden_context_with_string_content():
    items = [
        HiddenContextItem(
            id="123",
            content="User pressed the red button",
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
        )
    ]

    # The default converter works for string content
    items = await simple_to_agent_input(items)
    assert len(items) == 1
    assert items[0] == {
        "content": [
            {
                "text": "Hidden context for the agent (not shown to the user):\n<HiddenContext>\nUser pressed the red button\n</HiddenContext>",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }


async def test_input_item_converter_for_hidden_context_with_non_string_content():
    items = [
        HiddenContextItem(
            id="123",
            content={"harry": "potter", "hermione": "granger"},
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
        )
    ]

    # Default converter should throw
    with pytest.raises(NotImplementedError):
        await simple_to_agent_input(items)

    class MyThreadItemConverter(ThreadItemConverter):
        async def hidden_context_to_input(self, item: HiddenContextItem):
            content = ",".join(item.content.keys())
            return Message(
                type="message",
                content=[
                    ResponseInputTextParam(
                        type="input_text", text=f"<HIDDEN>{content}</HIDDEN>"
                    )
                ],
                role="user",
            )

    items = await MyThreadItemConverter().to_agent_input(items)
    assert len(items) == 1
    assert items[0] == {
        "content": [
            {
                "text": "<HIDDEN>harry,hermione</HIDDEN>",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }


async def test_input_item_converter_with_client_tool_call():
    items = [
        UserMessageItem(
            id="123",
            content=[UserMessageTextContent(text="Call a client tool call xyz")],
            attachments=[],
            inference_options=InferenceOptions(),
            thread_id=_THREAD_ID,
            quoted_text="Hi!",
            created_at=datetime.now(),
        ),
        TaskItem(
            id="tsk_123",
            created_at=datetime.now(),
            task=CustomTask(title="Called xyx"),
            thread_id=_THREAD_ID,
        ),
        ClientToolCallItem(
            id="ctc_123",
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
            name="xyz",
            arguments={"foo": "bar"},
            call_id="ctc_123",
        ),
        ClientToolCallItem(
            id="ctc_123_done",
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
            name="xyz",
            arguments={"foo": "bar"},
            call_id="ctc_123",
            status="completed",
            output={"success": True},
        ),
    ]

    input_items = await simple_to_agent_input(items)
    assert len(input_items) == 2
    assert input_items[0] == {
        "content": [
            {
                "text": "Call a client tool call xyz",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }
    assert input_items[1] == {
        "content": [
            {
                "text": "A message was displayed to the user that the following task was performed:\n<Task>\nCalled xyx\n</Task>",
                "type": "input_text",
            },
        ],
        "type": "message",
        "role": "user",
    }


def test_client_tool_continuation_replaces_only_the_deferred_output():
    call = ClientToolCallItem(
        id="tc_example",
        thread_id=_THREAD_ID,
        created_at=datetime.now(),
        name="browser_tool",
        arguments={"browser_only": True},
        call_id="call_example",
    )
    with pytest.raises(ValueError, match="Complete"):
        client_tool_continuation_input(call)

    completed = call.model_copy(update={"status": "completed", "output": None})
    with pytest.raises(ValueError, match="no captured native model continuation"):
        client_tool_continuation_input(completed)

    deferred: dict[str, JsonValue] = {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": "None",
    }
    sibling: dict[str, JsonValue] = {
        "type": "function_call_output",
        "call_id": "call_sibling",
        "output": "server result",
    }
    completed.model_continuation = ClientToolLocalContinuation(
        items=[sibling, deferred]
    )
    assert client_tool_continuation_input(completed) == [
        sibling,
        {**deferred, "output": "null"},
    ]
    assert completed.model_continuation.items == [sibling, deferred]

    for invalid_items in ([sibling], [deferred, deferred]):
        completed.model_continuation = ClientToolLocalContinuation(items=invalid_items)
        with pytest.raises(ValueError, match="exactly one matching output"):
            client_tool_continuation_input(completed)


async def test_input_item_converter_with_structured_input():
    items = [
        StructuredInputItem(
            id="si_123",
            thread_id=_THREAD_ID,
            created_at=datetime.now(),
            status="answered",
            inputs=[
                StructuredInputMultipleChoice(
                    id="subject",
                    question="Which subject is this lesson for?",
                    options=[
                        StructuredInputMultipleChoiceOption(value="Math"),
                        StructuredInputMultipleChoiceOption(value="Science"),
                    ],
                    answer=StructuredInputAnswer(values=["Math"]),
                ),
                StructuredInputFreeform(
                    id="details",
                    question="Anything else to know?",
                    answer=StructuredInputAnswer(skipped=True),
                ),
            ],
        )
    ]

    input_items = await simple_to_agent_input(items)
    assert len(input_items) == 1
    assert input_items[0] == {
        "content": [
            {
                "text": "A structured input request was displayed to the user with the following status: answered\n<StructuredInput>\n- Which subject is this lesson for?: Math\n- Anything else to know?: skipped\n</StructuredInput>",
                "type": "input_text",
            },
        ],
        "role": "user",
        "type": "message",
    }

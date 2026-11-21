from datetime import UTC, datetime
from typing import Any, AsyncIterator

from chatkit.actions import Action, IncomingAction
from chatkit.events import (
    SyncCustomActionResponse,
    ThreadCreatedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
    WidgetRootUpdated,
)
from chatkit.models import (
    AssistantMessageContent,
    AssistantMessageItem,
    ClientToolCallItem,
    InferenceOptions,
    StructuredInputFreeform,
    StructuredInputItem,
    StructuredInputMultipleChoice,
    StructuredInputMultipleChoiceOption,
    ThreadMetadata,
    UserMessageInput,
    UserMessageTextContent,
    WidgetItem,
)
from chatkit.requests import (
    ClientToolCallbackResult,
    ItemFeedbackParams,
    ItemsFeedbackReq,
    StructuredInputAnswerSubmission,
    StructuredInputSubmission,
    ThreadAddClientToolOutputParams,
    ThreadAddStructuredInputParams,
    ThreadCreateParams,
    ThreadCustomActionParams,
    ThreadsAddClientToolOutputReq,
    ThreadsAddStructuredInputReq,
    ThreadsCreateReq,
    ThreadsCustomActionReq,
    ThreadsSyncCustomActionReq,
)
from chatkit.turns import (
    ClientToolTurn,
    StructuredInputTurn,
    Turn,
    UserMessageTurn,
)
from chatkit.widgets import Card, Text
from tests._types import RequestContext
from tests.helpers.server import (
    DEFAULT_CONTEXT,
    make_server,
)


async def test_respond_with_tool_call():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        if isinstance(turn.input, UserMessageTurn):
            yield ThreadItemDoneEvent(
                item=ClientToolCallItem(
                    id="msg_1",
                    created_at=datetime.now(UTC),
                    name="tool_call_1",
                    arguments={"arg1": "val1", "arg2": False},
                    call_id="tool_call_1",
                    thread_id=thread.id,
                ),
            )
        elif isinstance(turn.input, ClientToolTurn):
            yield ThreadItemDoneEvent(
                item=AssistantMessageItem(
                    id="msg_2",
                    content=[
                        AssistantMessageContent(text="Glad the tool call succeeded!")
                    ],
                    created_at=datetime.now(UTC),
                    thread_id=thread.id,
                ),
            )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Hello, world!")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )

        assert len(events) == 4
        assert events[0].type == "thread.created"
        thread = events[0].thread

        assert events[1].type == "thread.item.done"
        assert events[1].item.type == "user_message"

        assert events[2].type == "stream_options"

        assert events[3].type == "thread.item.done"
        assert events[3].item.type == "client_tool_call"
        assert events[3].item.id == "msg_1"
        assert events[3].item.name == "tool_call_1"
        assert events[3].item.arguments == {"arg1": "val1", "arg2": False}
        assert events[3].item.call_id == "tool_call_1"

        events = await server.process_streaming(
            ThreadsAddClientToolOutputReq(
                params=ThreadAddClientToolOutputParams(
                    thread_id=thread.id,
                    result=ClientToolCallbackResult(
                        item_id="msg_1", output={"text": "Wow!"}
                    ),
                )
            )
        )

        assert len(events) == 2
        assert events[0].type == "stream_options"
        assert events[1].type == "thread.item.done"
        assert events[1].item.type == "assistant_message"


async def test_calls_action():
    actions = []

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=WidgetItem(
                id="widget_1",
                type="widget",
                created_at=datetime.now(UTC),
                thread_id=thread.id,
                widget=Card(
                    children=[
                        Text(
                            key="text_1",
                            value="test",
                        )
                    ],
                ).build(),
            ),
        )

    async def action(
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: Any,
    ) -> AsyncIterator[ThreadStreamEvent]:
        actions.append((action, sender))
        assert sender

        yield ThreadItemUpdatedEvent(
            item_id=sender.id,
            update=WidgetRootUpdated(
                widget=Card(
                    children=[
                        Text(value="Email sent!"),
                    ]
                ).build(),
            ),
        )

    with make_server(responder, action_callback=action) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Show widget")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        widget_item = next(
            event.item
            for event in events
            if isinstance(event, ThreadItemDoneEvent)
            and isinstance(event.item, WidgetItem)
        )

        events = await server.process_streaming(
            ThreadsCustomActionReq(
                params=ThreadCustomActionParams(
                    thread_id=thread.id,
                    item_id=widget_item.id,
                    action=IncomingAction(
                        type="create_user", payload={"user_id": "123"}
                    ),
                )
            )
        )

        assert actions
        assert actions[0] == (
            IncomingAction(type="create_user", payload={"user_id": "123"}),
            widget_item,
        )

        assert len(events) == 2
        assert events[0].type == "stream_options"
        assert events[1].type == "thread.item.updated"
        assert isinstance(events[1], ThreadItemUpdatedEvent)
        assert events[1].update.type == "widget.root.updated"
        assert (
            events[1].update.widget
            == Card(children=[Text(value="Email sent!")]).build()
        )


async def test_add_structured_input_replaces_item_with_custom_answer_and_resumes_response():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        if isinstance(turn.input, UserMessageTurn):
            yield ThreadItemDoneEvent(
                item=StructuredInputItem(
                    id="si_1",
                    thread_id=thread.id,
                    created_at=datetime.now(UTC),
                    inputs=[
                        StructuredInputMultipleChoice(
                            id="subject",
                            question="Which subject is this lesson for?",
                            options=[
                                StructuredInputMultipleChoiceOption(value="Math"),
                                StructuredInputMultipleChoiceOption(value="Science"),
                            ],
                        ),
                        StructuredInputFreeform(
                            id="details",
                            question="Anything else to know?",
                        ),
                    ],
                )
            )
            return

        assert isinstance(turn.input, StructuredInputTurn)
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="msg_after_structured_input",
                content=[AssistantMessageContent(text="Thanks, continuing now.")],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            )
        )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Plan a lesson")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if isinstance(event, ThreadCreatedEvent)
        )

        events = await server.process_streaming(
            ThreadsAddStructuredInputReq(
                params=ThreadAddStructuredInputParams(
                    thread_id=thread.id,
                    item_id="si_1",
                    input=StructuredInputSubmission(
                        status="answered",
                        answers={
                            "subject": StructuredInputAnswerSubmission(
                                values=["Private class"]
                            ),
                            "details": StructuredInputAnswerSubmission(
                                values=["Please include group work."]
                            ),
                        },
                    ),
                )
            )
        )

        replaced_event = next(
            event for event in events if isinstance(event, ThreadItemReplacedEvent)
        )
        assert isinstance(replaced_event.item, StructuredInputItem)
        assert replaced_event.item.status == "answered"
        assert replaced_event.item.inputs[0].answer is not None
        assert replaced_event.item.inputs[0].answer.values == ["Private class"]
        assert replaced_event.item.inputs[1].answer is not None
        assert replaced_event.item.inputs[1].answer.values == [
            "Please include group work."
        ]

        persisted_item = await server.store.load_item(
            thread.id,
            "si_1",
            DEFAULT_CONTEXT,
        )
        assert persisted_item == replaced_event.item
        assert any(
            isinstance(event, ThreadItemDoneEvent)
            and isinstance(event.item, AssistantMessageItem)
            and event.item.id == "msg_after_structured_input"
            for event in events
        )


async def test_add_structured_input_treats_omitted_answers_as_skipped():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        if isinstance(turn.input, UserMessageTurn):
            yield ThreadItemDoneEvent(
                item=StructuredInputItem(
                    id="si_1",
                    thread_id=thread.id,
                    created_at=datetime.now(UTC),
                    inputs=[
                        StructuredInputMultipleChoice(
                            id="subject",
                            question="Which subject is this lesson for?",
                            options=[
                                StructuredInputMultipleChoiceOption(value="Math"),
                                StructuredInputMultipleChoiceOption(value="Science"),
                            ],
                        ),
                        StructuredInputFreeform(
                            id="details",
                            question="Anything else to know?",
                        ),
                    ],
                )
            )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Plan a lesson")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if isinstance(event, ThreadCreatedEvent)
        )

        events = await server.process_streaming(
            ThreadsAddStructuredInputReq(
                params=ThreadAddStructuredInputParams(
                    thread_id=thread.id,
                    item_id="si_1",
                    input=StructuredInputSubmission(
                        status="answered",
                        answers={
                            "subject": StructuredInputAnswerSubmission(values=["Math"]),
                        },
                    ),
                )
            )
        )

        replaced_event = next(
            event for event in events if isinstance(event, ThreadItemReplacedEvent)
        )
        assert isinstance(replaced_event.item, StructuredInputItem)
        subject_answer = replaced_event.item.inputs[0].answer
        assert subject_answer is not None
        assert subject_answer.values == ["Math"]
        details_answer = replaced_event.item.inputs[1].answer
        assert details_answer is not None
        assert details_answer.skipped


async def test_add_structured_input_ignores_unknown_answer_ids():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        if isinstance(turn.input, UserMessageTurn):
            yield ThreadItemDoneEvent(
                item=StructuredInputItem(
                    id="si_1",
                    thread_id=thread.id,
                    created_at=datetime.now(UTC),
                    inputs=[
                        StructuredInputFreeform(
                            id="details",
                            question="Anything else to know?",
                        ),
                    ],
                )
            )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Plan a lesson")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if isinstance(event, ThreadCreatedEvent)
        )

        events = await server.process_streaming(
            ThreadsAddStructuredInputReq(
                params=ThreadAddStructuredInputParams(
                    thread_id=thread.id,
                    item_id="si_1",
                    input=StructuredInputSubmission(
                        status="answered",
                        answers={
                            "details": StructuredInputAnswerSubmission(
                                values=["Please include group work."]
                            ),
                            "unknown": StructuredInputAnswerSubmission(
                                values=["ignored"]
                            ),
                        },
                    ),
                )
            )
        )

        replaced_event = next(
            event for event in events if isinstance(event, ThreadItemReplacedEvent)
        )
        assert isinstance(replaced_event.item, StructuredInputItem)
        answer = replaced_event.item.inputs[0].answer
        assert answer is not None
        assert answer.values == ["Please include group work."]


async def test_add_structured_input_truncates_extra_single_choice_values():
    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=StructuredInputItem(
                id="si_1",
                thread_id=thread.id,
                created_at=datetime.now(UTC),
                inputs=[
                    StructuredInputMultipleChoice(
                        id="subject",
                        question="Which subject is this lesson for?",
                        options=[
                            StructuredInputMultipleChoiceOption(value="Math"),
                            StructuredInputMultipleChoiceOption(value="Science"),
                        ],
                    ),
                ],
            )
        )

    with make_server(responder) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Plan a lesson")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if isinstance(event, ThreadCreatedEvent)
        )

        events = await server.process_streaming(
            ThreadsAddStructuredInputReq(
                params=ThreadAddStructuredInputParams(
                    thread_id=thread.id,
                    item_id="si_1",
                    input=StructuredInputSubmission(
                        status="answered",
                        answers={
                            "subject": StructuredInputAnswerSubmission(
                                values=["Math", "Science"],
                            ),
                        },
                    ),
                )
            )
        )

        replaced_event = next(
            event for event in events if isinstance(event, ThreadItemReplacedEvent)
        )
        assert isinstance(replaced_event.item, StructuredInputItem)
        answer = replaced_event.item.inputs[0].answer
        assert answer is not None
        assert answer.values == ["Math"]

        persisted_item = await server.store.load_item(
            thread.id,
            "si_1",
            DEFAULT_CONTEXT,
        )
        assert isinstance(persisted_item, StructuredInputItem)
        assert persisted_item.status == "answered"
        assert persisted_item.inputs[0].answer == answer


async def test_calls_action_sync():
    sync_actions = []

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=WidgetItem(
                id="widget_1",
                type="widget",
                created_at=datetime.now(UTC),
                thread_id=thread.id,
                widget=Card(
                    children=[
                        Text(
                            key="text_1",
                            value="test",
                        )
                    ],
                ).build(),
            ),
        )

    def sync_action(
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: Any,
    ) -> SyncCustomActionResponse:
        sync_actions.append((action, sender))
        assert sender

        return SyncCustomActionResponse(
            updated_item=sender.model_copy(
                update={
                    "widget": Card(
                        children=[
                            Text(value="Email sent!"),
                        ]
                    ).build()
                }
            )
        )

    with make_server(responder, sync_action_callback=sync_action) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Show widget")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )
        widget_item = next(
            event.item
            for event in events
            if isinstance(event, ThreadItemDoneEvent)
            and isinstance(event.item, WidgetItem)
        )

        response = await server.process_non_streaming(
            ThreadsSyncCustomActionReq(
                params=ThreadCustomActionParams(
                    thread_id=thread.id,
                    item_id=widget_item.id,
                    action=IncomingAction(
                        type="create_user", payload={"user_id": "123"}
                    ),
                )
            )
        )
        assert isinstance(response, SyncCustomActionResponse)

        assert sync_actions
        assert sync_actions[0] == (
            IncomingAction(type="create_user", payload={"user_id": "123"}),
            widget_item,
        )
        assert response.updated_item == widget_item.model_copy(
            update={"widget": Card(children=[Text(value="Email sent!")]).build()}
        )


async def test_add_feedback():
    called = []

    def handle_feedback(thread_id, item_ids, feedback, context):
        called.append((thread_id, item_ids, feedback, context))

    async def responder(turn: Turn[RequestContext]) -> AsyncIterator[ThreadStreamEvent]:
        thread = turn.thread
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id="assistant_msg_1",
                content=[
                    AssistantMessageContent(text="Feedback received!"),
                ],
                created_at=datetime.now(UTC),
                thread_id=thread.id,
            ),
        )
        yield ThreadItemDoneEvent(
            item=WidgetItem(
                id="widget_1",
                type="widget",
                created_at=datetime.now(UTC),
                widget=Card(
                    children=[Text(key="text", value="Widget content")]
                ).build(),
                thread_id=thread.id,
            ),
        )

    with make_server(responder, handle_feedback=handle_feedback) as server:
        events = await server.process_streaming(
            ThreadsCreateReq(
                params=ThreadCreateParams(
                    input=UserMessageInput(
                        content=[UserMessageTextContent(text="Test feedback")],
                        attachments=[],
                        inference_options=InferenceOptions(),
                    )
                )
            )
        )
        thread = next(
            event.thread for event in events if event.type == "thread.created"
        )

        await server.process_non_streaming(
            ItemsFeedbackReq(
                params=ItemFeedbackParams(
                    thread_id=thread.id,
                    item_ids=["assistant_msg_1"],
                    kind="positive",
                )
            )
        )
        await server.process_non_streaming(
            ItemsFeedbackReq(
                params=ItemFeedbackParams(
                    thread_id=thread.id,
                    item_ids=["widget_1"],
                    kind="negative",
                )
            )
        )

        assert len(called) == 2
        assert called[0][0] == thread.id
        assert called[0][1] == ["assistant_msg_1"]
        assert called[0][2] == "positive"

        assert called[1][0] == thread.id
        assert called[1][1] == ["widget_1"]
        assert called[1][2] == "negative"

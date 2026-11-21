"""Synthetic SDK event stream and ChatKit context setup for adapter tests."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal
from unittest.mock import AsyncMock, Mock

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    InputGuardrailResult,
    InputGuardrailTripwireTriggered,
    OutputGuardrail,
    OutputGuardrailResult,
    OutputGuardrailTripwireTriggered,
    RunResultStreaming,
    StreamEvent,
)

from chatkit.models import (
    Page,
    Thread,
)
from chatkit.turns import ResponderEvent


def make_thread_and_store() -> tuple[Thread, Mock]:
    thread = Thread(id="123", title="Test", created_at=datetime.now(), items=Page())
    store = Mock()
    store.generate_item_id = lambda item_type, thread, context: f"{item_type}_id"
    store.load_thread_items = AsyncMock(return_value=Page())
    store.add_thread_item = AsyncMock()
    return thread, store


class RunResult(RunResultStreaming):
    """Provide the stream methods used by ChatKit with synthetic events."""

    def __init__(self) -> None:
        self.is_complete = False
        self.interruptions = []
        self._test_events: asyncio.Queue[StreamEvent | Exception | None] = (
            asyncio.Queue()
        )

    def add_event(self, event: StreamEvent) -> None:
        self._test_events.put_nowait(event)

    def done(self, error: Exception | None = None) -> None:
        self.is_complete = True
        self._test_events.put_nowait(error)

    def cancel(self, mode: Literal["immediate", "after_turn"] = "immediate") -> None:
        self.done()

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        while (event := await self._test_events.get()) is not None:
            if isinstance(event, Exception):
                raise event
            yield event

    def throw_input_guardrails(self) -> None:
        self.done(
            InputGuardrailTripwireTriggered(
                InputGuardrailResult(
                    guardrail=Mock(spec=InputGuardrail),
                    output=GuardrailFunctionOutput(
                        output_info=None,
                        tripwire_triggered=True,
                    ),
                )
            )
        )

    def throw_output_guardrails(self) -> None:
        self.done(
            OutputGuardrailTripwireTriggered(
                OutputGuardrailResult(
                    guardrail=Mock(spec=OutputGuardrail),
                    output=GuardrailFunctionOutput(
                        output_info=None,
                        tripwire_triggered=True,
                    ),
                    agent=Mock(spec=Agent),
                    agent_output=None,
                )
            )
        )


async def all_events(
    events: AsyncIterator[ResponderEvent],
) -> list[ResponderEvent]:
    return [event async for event in events]

"""Typed inputs and outcomes for a server turn."""

from dataclasses import dataclass
from enum import StrEnum

from .events import ThreadStreamEvent
from .models import (
    ClientToolCallItem,
    StructuredInputItem,
    ThreadMetadata,
    UserMessageItem,
    WorkflowItem,
)


@dataclass(frozen=True, slots=True)
class PersistWorkflow:
    """Save still-open workflow progress without sending a browser event."""

    item: WorkflowItem
    replace: bool


type ResponderEvent = ThreadStreamEvent | PersistWorkflow


@dataclass(frozen=True, slots=True)
class UserMessageTurn:
    """A user message saved to a new or existing thread."""

    message: UserMessageItem


@dataclass(frozen=True, slots=True)
class ClientToolTurn:
    """A continuation after a client tool result was saved."""

    tool_call: ClientToolCallItem


@dataclass(frozen=True, slots=True)
class StructuredInputTurn:
    """A continuation after a structured input was answered or skipped."""

    item: StructuredInputItem


@dataclass(frozen=True, slots=True)
class RetryTurn:
    """A retry from an existing user message after removing later items."""

    message: UserMessageItem


type TurnInput = UserMessageTurn | ClientToolTurn | StructuredInputTurn | RetryTurn


@dataclass(frozen=True, slots=True)
class Turn[TContext]:
    """A durable turn ID, thread snapshot, trigger, and request context.

    Persist thread changes with an explicit patch.
    """

    thread: ThreadMetadata
    input: TurnInput
    context: TContext
    id: str


class TurnOutcome(StrEnum):
    """The result of processing a turn on the server."""

    COMPLETED = "completed"
    WAITING = "waiting"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class TurnState(StrEnum):
    """Durable server milestones for one thread's turn."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ABANDONED = "abandoned"
    SUPERSEDED = "superseded"

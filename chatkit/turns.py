"""Typed inputs and outcomes for a server turn."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

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


class ClientToolLocalContinuation(BaseModel):
    """Native model input to replay after a browser tool returns."""

    type: Literal["local"] = "local"
    items: list[dict[str, JsonValue]]


class ClientToolResponsesContinuation(BaseModel):
    """New function outputs and their predecessor in OpenAI-managed context."""

    type: Literal["responses"] = "responses"
    response_id: str = Field(min_length=1)
    items: list[dict[str, JsonValue]]


type ClientToolModelContinuation = Annotated[
    ClientToolLocalContinuation | ClientToolResponsesContinuation,
    Field(discriminator="type"),
]


class ClientToolInvocation(BaseModel):
    """Private execution record stored separately from the browser item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    thread_id: str
    turn_id: str
    call_id: str = Field(min_length=1)
    model_continuation: ClientToolModelContinuation | None = Field(
        default=None, repr=False
    )
    continuation_turn_id: str | None = None


@dataclass(frozen=True, slots=True)
class ClientToolRequest:
    """Ask the server to register a browser item and its private invocation."""

    item: ClientToolCallItem
    call_id: str
    model_continuation: ClientToolModelContinuation | None = field(
        default=None, repr=False
    )


@dataclass(frozen=True, slots=True)
class ApprovalRequired:
    """Private run checkpoint awaiting application review."""

    checkpoint: str = field(repr=False)


type ResponderEvent = (
    ThreadStreamEvent | PersistWorkflow | ClientToolRequest | ApprovalRequired
)


@dataclass(frozen=True, slots=True)
class UserMessageTurn:
    """A user message saved to a new or existing thread."""

    message: UserMessageItem


@dataclass(frozen=True, slots=True)
class ClientToolTurn:
    """A continuation after a client tool result was saved."""

    tool_call: ClientToolCallItem
    invocation: ClientToolInvocation


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

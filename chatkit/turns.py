"""Typed inputs and outcomes for a server turn."""

from dataclasses import dataclass
from enum import StrEnum

from .types import (
    ClientToolCallItem,
    StructuredInputItem,
    ThreadMetadata,
    UserMessageItem,
)


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
    """A turn's thread, trigger, and app-supplied request context."""

    thread: ThreadMetadata
    input: TurnInput
    context: TContext


class TurnOutcome(StrEnum):
    """The result of processing a turn on the server."""

    COMPLETED = "completed"
    WAITING = "waiting"
    CANCELLED = "cancelled"
    FAILED = "failed"

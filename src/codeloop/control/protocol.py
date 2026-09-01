"""Pure Human Interaction protocol shared across Agent and Interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

InteractionKind = Literal[
    "inform",
    "approve",
    "re_approve",
    "clarify",
    "choose",
]
InteractionResponseStatus = Literal["answered", "unavailable", "interrupted"]


@dataclass(frozen=True)
class InteractionOption:
    id: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class InteractionAction:
    """A bounded public description of the action under discussion."""

    description: str
    category: str | None = None
    command: tuple[str, ...] = ()
    cwd: str | None = None
    authorization_basis: str | None = None
    workspace_root: str | None = None
    previous_command: tuple[str, ...] = ()
    previous_cwd: str | None = None
    scope_change: str | None = None


@dataclass(frozen=True)
class InteractionRequest:
    kind: InteractionKind
    prompt: str
    options: tuple[InteractionOption, ...] = ()
    action: InteractionAction | None = None


@dataclass(frozen=True)
class InteractionResponse:
    status: InteractionResponseStatus
    answer: str | None = None
    selected_option_id: str | None = None
    approved: bool | None = None


class InteractionProvider(Protocol):
    """Narrow boundary through which Agent Core returns control to a human."""

    def interact(self, request: InteractionRequest) -> InteractionResponse:
        ...

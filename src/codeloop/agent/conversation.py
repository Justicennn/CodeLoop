"""Public, task-to-task conversation data owned by Agent Core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicConversationTurn:
    """One bounded public exchange supplied as context for a later task.

    This intentionally excludes tool cycles and all private runtime state.
    """

    user: str
    assistant: str

    def __post_init__(self) -> None:
        if not isinstance(self.user, str) or not isinstance(self.assistant, str):
            raise TypeError("Public conversation text must be strings")

"""Deterministic, bounded conversation history for the agent loop."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

Message = dict[str, Any]

DEFAULT_MAX_CONTEXT_CHARS = 80_000
DEFAULT_MAX_CONTEXT_MESSAGES = 40
MIN_CONTEXT_CHARS = 1_000
MAX_CONTEXT_CHARS = 1_000_000
MIN_CONTEXT_MESSAGES = 5
MAX_CONTEXT_MESSAGES = 200

_NOTICE_PREFIX = "Runtime context notice: "


class ConversationContext:
    """Keep pinned messages and protocol-complete tool cycles within budgets."""

    def __init__(
        self,
        system_prompt: str,
        task: str,
        *,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
    ) -> None:
        _validate_budget(
            "max_context_chars",
            max_chars,
            MIN_CONTEXT_CHARS,
            MAX_CONTEXT_CHARS,
        )
        _validate_budget(
            "max_context_messages",
            max_messages,
            MIN_CONTEXT_MESSAGES,
            MAX_CONTEXT_MESSAGES,
        )
        self._system_message: Message = {
            "role": "system",
            "content": system_prompt,
        }
        self._task_message: Message = {"role": "user", "content": task}
        self._max_chars = max_chars
        self._max_messages = max_messages
        self._cycles: list[list[Message]] = []
        self._removed_cycles = 0
        self._removed_messages = 0
        self._overflow = False
        self._rebalance()

    def add_tool_cycle(
        self,
        assistant_message: Message,
        tool_messages: list[Message],
    ) -> None:
        """Append one complete assistant-call/result cycle, then trim old cycles."""
        call_ids = _assistant_call_ids(assistant_message)
        result_ids = _tool_result_ids(tool_messages)
        if not call_ids or result_ids != call_ids:
            raise ValueError(
                "A tool cycle must contain one result for every assistant tool call"
            )
        self._cycles.append(deepcopy([assistant_message, *tool_messages]))
        self._rebalance()

    def messages_for_model(self) -> list[Message]:
        """Return a snapshot containing only pinned messages and complete cycles."""
        return deepcopy(self._compose_messages())

    def _rebalance(self) -> None:
        self._overflow = False
        while not self._within_budget(self._compose_messages()):
            if len(self._cycles) <= 1:
                self._overflow = True
                return
            removed = self._cycles.pop(0)
            self._removed_cycles += 1
            self._removed_messages += len(removed)

    def _compose_messages(self) -> list[Message]:
        messages = [self._system_message]
        if self._removed_cycles or self._overflow:
            messages.append(self._notice_message())
        messages.append(self._task_message)
        for cycle in self._cycles:
            messages.extend(cycle)
        return messages

    def _notice_message(self) -> Message:
        metadata = {
            "conversation_history_trimmed": self._removed_cycles > 0,
            "guidance": "Older tool evidence is unavailable; re-read files if needed.",
            "overflow": self._overflow,
            "removed_cycles": self._removed_cycles,
            "removed_messages": self._removed_messages,
        }
        return {
            "role": "system",
            "content": _NOTICE_PREFIX
            + json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    def _within_budget(self, messages: list[Message]) -> bool:
        return (
            len(messages) <= self._max_messages
            and _serialized_char_count(messages) <= self._max_chars
        )


def _serialized_char_count(messages: list[Message]) -> int:
    return len(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _assistant_call_ids(message: Message) -> list[str]:
    if message.get("role") != "assistant":
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    ids: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str):
            return []
        ids.append(call["id"])
    return ids


def _tool_result_ids(messages: list[Message]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            return []
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            return []
        ids.append(tool_call_id)
    return ids


def _validate_budget(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")

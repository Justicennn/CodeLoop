"""The explicit Stage 1 decision-action-observation agent loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .llm import ModelClient, ToolCall
from .tools import ToolRegistry, ToolResult

SYSTEM_PROMPT = """You are an agent that can use tools to interact with a local runtime.
Use a tool when it helps complete the user's task. Continue from the returned tool result.
Never pretend a tool ran. If a tool fails, use the observation to correct your next decision."""

ToolEventHandler = Callable[[ToolCall, ToolResult], None]


@dataclass(frozen=True)
class AgentResult:
    status: Literal["completed", "max_steps"]
    answer: str | None
    steps: int


class AgentRunner:
    """Own conversation history, local actions, observations, and stopping."""

    def __init__(
        self,
        client: ModelClient,
        *,
        tools: ToolRegistry | None = None,
        max_steps: int = 20,
        on_tool_event: ToolEventHandler | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._client = client
        self._tools = tools or ToolRegistry()
        self._max_steps = max_steps
        self._on_tool_event = on_tool_event

    def run(self, task: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for step in range(1, self._max_steps + 1):
            response = self._client.complete(messages, self._tools.schemas)

            if not response.tool_calls:
                return AgentResult(status="completed", answer=response.text or "", steps=step)

            messages.append(self._assistant_tool_call_message(response.text, response.tool_calls))

            for tool_call in response.tool_calls:
                result = self._tools.dispatch(tool_call.name, tool_call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if self._on_tool_event is not None:
                    self._on_tool_event(tool_call, result)

        return AgentResult(status="max_steps", answer=None, steps=self._max_steps)

    @staticmethod
    def _assistant_tool_call_message(
        text: str | None,
        tool_calls: list[ToolCall],
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in tool_calls
            ],
        }

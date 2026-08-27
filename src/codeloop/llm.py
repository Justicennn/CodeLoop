"""One narrow OpenAI-compatible model API path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI


@dataclass(frozen=True)
class ToolCall:
    """A provider-independent native tool call used by the runner."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelResponse:
    """The small response shape understood by the agent loop."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelClient(Protocol):
    """Structural contract shared by the real client and test fake."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return the model's next normalized decision."""


class OpenAICompatibleClient:
    """Send chat/tool-calling requests through one ordinary API client."""

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        if not api_key or not base_url or not model:
            raise ValueError("MODEL_API_KEY, MODEL_BASE_URL, and MODEL_NAME are required")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        normalized_calls: list[ToolCall] = []

        for index, tool_call in enumerate(message.tool_calls or []):
            call_id = tool_call.id or f"missing_tool_call_id_{index}"
            normalized_calls.append(
                ToolCall(
                    id=call_id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            )

        return ModelResponse(text=message.content, tool_calls=normalized_calls)

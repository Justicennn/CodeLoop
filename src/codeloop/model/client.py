"""One narrow OpenAI-compatible model API path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
)

APIErrorClassification = Literal["retryable", "fatal"]


class ModelAPIError(Exception):
    """A provider error reduced to safe retry semantics."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        classification: APIErrorClassification,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.safe_message = message
        self.classification = classification


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

    @property
    def supports_image_input(self) -> bool:
        """Whether the caller explicitly enabled native image input."""
        ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return the model's next normalized decision."""


class OpenAICompatibleClient:
    """Send chat/tool-calling requests through one ordinary API client."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        supports_image_input: bool = False,
    ) -> None:
        if not api_key or not base_url or not model:
            raise ValueError("MODEL_API_KEY, MODEL_BASE_URL, and MODEL_NAME are required")
        if not isinstance(supports_image_input, bool):
            raise ValueError("supports_image_input must be a boolean")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )
        self._model = model
        self._supports_image_input = supports_image_input

    @property
    def supports_image_input(self) -> bool:
        return self._supports_image_input

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise ModelAPIError(
                "temporary_api_error",
                "The model API is temporarily unavailable.",
                classification="retryable",
            ) from exc
        except APIStatusError as exc:
            status_code = exc.status_code
            if status_code == 429 or status_code >= 500:
                raise ModelAPIError(
                    "temporary_api_error",
                    "The model API returned a temporary error.",
                    classification="retryable",
                ) from exc
            raise ModelAPIError(
                "model_api_error",
                "The model API rejected the request or configuration.",
                classification="fatal",
            ) from exc
        except OpenAIError as exc:
            raise ModelAPIError(
                "model_api_error",
                "The model API request failed.",
                classification="fatal",
            ) from exc

        if not response.choices:
            raise ModelAPIError(
                "invalid_model_response",
                "The model API returned no response choice.",
                classification="fatal",
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

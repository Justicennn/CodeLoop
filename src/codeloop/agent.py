"""The explicit decision-action-observation agent loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Literal

from .context import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_MESSAGES,
    ConversationContext,
)
from .llm import ModelAPIError, ModelClient, ModelResponse, ToolCall
from .tools import ToolRegistry, ToolResult

SYSTEM_PROMPT = """You are an agent that can use tools to interact with a local runtime.
For coding tasks, inspect the relevant existing files and establish a baseline before editing when practical.
Use tool and command failures as evidence: diagnose them, make a focused repair, and run relevant verification after changes.
Never pretend a tool or check ran, and never claim success that the observations do not support.
In the final answer, honestly name changed files, verification commands and results, and any known limitations."""

TerminationReason = Literal[
    "completed",
    "max_steps",
    "repeated_failure",
    "fatal_api_error",
    "user_interrupt",
    "runtime_error",
]
MODEL_REQUEST_ATTEMPTS = 3
MODEL_RETRY_DELAYS = (0.5, 1.0)
MAX_CONFIGURED_STEPS = 100


@dataclass(frozen=True)
class AgentResult:
    status: TerminationReason
    answer: str | None
    steps: int
    message: str | None = None


@dataclass(frozen=True)
class ToolEvent:
    tool_call: ToolCall
    result: ToolResult
    dispatch_duration_ms: int
    truncated: bool


ToolEventHandler = Callable[[ToolEvent], None]


class _FailureTracker:
    """Track only consecutive, identical failed tool calls."""

    def __init__(self) -> None:
        self._fingerprint: str | None = None
        self._count = 0

    def record(self, tool_call: ToolCall, result: ToolResult) -> bool:
        if result.get("ok") is True:
            self._fingerprint = None
            self._count = 0
            return False
        if result.get("ok") is not False:
            raise ValueError("Tool result must contain a boolean ok field")

        error_code = result.get("error_code")
        if not isinstance(error_code, str):
            raise ValueError("Failed tool result must contain an error_code")
        fingerprint = "\x00".join(
            (
                tool_call.name,
                self._canonical_arguments(tool_call.arguments),
                error_code,
            )
        )
        if fingerprint == self._fingerprint:
            self._count += 1
        else:
            self._fingerprint = fingerprint
            self._count = 1
        return self._count >= 3

    @staticmethod
    def _canonical_arguments(arguments: str) -> str:
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return f"raw:{arguments}"
        return json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class AgentRunner:
    """Own conversation history, local actions, observations, and stopping."""

    def __init__(
        self,
        client: ModelClient,
        *,
        tools: ToolRegistry,
        max_steps: int = 20,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_context_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
        on_tool_event: ToolEventHandler | None = None,
    ) -> None:
        if max_steps < 1 or max_steps > MAX_CONFIGURED_STEPS:
            raise ValueError(
                f"max_steps must be between 1 and {MAX_CONFIGURED_STEPS}"
            )
        self._client = client
        self._tools = tools
        self._max_steps = max_steps
        self._max_context_chars = max_context_chars
        self._max_context_messages = max_context_messages
        self._on_tool_event = on_tool_event

    def run(self, task: str) -> AgentResult:
        context = ConversationContext(
            SYSTEM_PROMPT,
            task,
            max_chars=self._max_context_chars,
            max_messages=self._max_context_messages,
        )
        failures = _FailureTracker()
        current_step = 0

        try:
            for step in range(1, self._max_steps + 1):
                current_step = step
                response = self._request_model(context.messages_for_model())

                if not response.tool_calls:
                    return AgentResult(
                        status="completed",
                        answer=response.text or "",
                        steps=step,
                    )

                assistant_message = self._assistant_tool_call_message(
                    response.text,
                    response.tool_calls,
                )
                tool_messages: list[dict[str, Any]] = []
                for tool_call in response.tool_calls:
                    dispatch_started = perf_counter()
                    result = self._tools.dispatch(
                        tool_call.name,
                        tool_call.arguments,
                    )
                    dispatch_duration_ms = max(
                        0,
                        round((perf_counter() - dispatch_started) * 1000),
                    )
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    if self._on_tool_event is not None:
                        self._on_tool_event(
                            ToolEvent(
                                tool_call=tool_call,
                                result=result,
                                dispatch_duration_ms=dispatch_duration_ms,
                                truncated=_tool_result_is_truncated(result),
                            )
                        )
                    if failures.record(tool_call, result):
                        return AgentResult(
                            status="repeated_failure",
                            answer=None,
                            steps=step,
                            message=(
                                "Three consecutive identical tool failures "
                                f"occurred for {tool_call.name}."
                            ),
                        )
                context.add_tool_cycle(assistant_message, tool_messages)

            return AgentResult(
                status="max_steps",
                answer=None,
                steps=self._max_steps,
                message=f"Maximum model decisions reached: {self._max_steps}.",
            )
        except KeyboardInterrupt:
            return AgentResult(
                status="user_interrupt",
                answer=None,
                steps=current_step,
                message="The run was interrupted by the user.",
            )
        except ModelAPIError as exc:
            return AgentResult(
                status="fatal_api_error",
                answer=None,
                steps=current_step,
                message=exc.safe_message,
            )
        except Exception:
            return AgentResult(
                status="runtime_error",
                answer=None,
                steps=current_step,
                message="An unexpected internal runtime error occurred.",
            )

    def _request_model(
        self,
        messages: list[dict[str, Any]],
    ) -> ModelResponse:
        for attempt in range(MODEL_REQUEST_ATTEMPTS):
            try:
                return self._client.complete(messages, self._tools.schemas)
            except ModelAPIError as exc:
                if exc.classification == "fatal":
                    raise
                if attempt == MODEL_REQUEST_ATTEMPTS - 1:
                    raise ModelAPIError(
                        "api_retry_exhausted",
                        "Temporary model API failures exhausted the retry budget.",
                        classification="retryable",
                    ) from exc
                sleep(MODEL_RETRY_DELAYS[attempt])
        raise RuntimeError("Model retry loop ended without a result")

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
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in tool_calls
            ],
        }


def _tool_result_is_truncated(result: ToolResult) -> bool:
    data = result.get("data")
    if not isinstance(data, dict):
        return False
    return any(
        data.get(field) is True
        for field in ("truncated", "stdout_truncated", "stderr_truncated")
    )

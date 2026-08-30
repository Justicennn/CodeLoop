"""The explicit decision-action-observation agent loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Literal

from ..execution.tools import ToolRegistry, ToolResult
from ..model.client import ModelAPIError, ModelClient, ModelResponse, ToolCall
from .context import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_MESSAGES,
    ConversationContext,
)
from .events import ModelRequestHandler, ToolEvent, ToolEventHandler
from .plan import (
    PlanStep,
    UPDATE_PLAN_ACTION_NAME,
    UPDATE_PLAN_SCHEMA,
    apply_plan_action,
)
from .prompt import SYSTEM_PROMPT
from .progress import ProgressAction, ProgressFacts, ProgressTracker
from .task_state import PlanOutcome, TaskState
from .verification import VerificationAttempt, VerificationStatus

TerminationReason = Literal[
    "completed",
    "max_steps",
    "repeated_failure",
    "no_progress",
    "fatal_api_error",
    "user_interrupt",
    "runtime_error",
]
MODEL_REQUEST_ATTEMPTS = 3
MODEL_RETRY_DELAYS = (0.5, 1.0)
MAX_CONFIGURED_STEPS = 100


@dataclass(frozen=True)
class AgentResult:
    """Termination plus orthogonal Plan and managed-revision verification facts.

    ``verification_status == "verified"`` means only that the latest
    run_command attempt for the current CodeLoop-managed revision succeeded.
    Command relevance, semantic correctness, and run_command filesystem side
    effects remain outside this result's guarantees.
    """

    status: TerminationReason
    answer: str | None
    steps: int
    message: str | None = None
    verification_status: VerificationStatus = "not_required"
    workspace_revision: int = 0
    verified_revision: int | None = None
    last_verification: VerificationAttempt | None = None
    plan_status: PlanOutcome = "not_planned"
    unfinished_steps: tuple[PlanStep, ...] = ()
    blocked_steps: tuple[PlanStep, ...] = ()


class _FailureTracker:
    """Track only consecutive, identical failed action/tool calls."""

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
        on_model_request_started: ModelRequestHandler | None = None,
        on_model_request_finished: ModelRequestHandler | None = None,
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
        self._on_model_request_started = on_model_request_started
        self._on_model_request_finished = on_model_request_finished
        execution_schemas = self._tools.schemas
        if UPDATE_PLAN_ACTION_NAME in self._tools.names or any(
            schema.get("function", {}).get("name") == UPDATE_PLAN_ACTION_NAME
            for schema in execution_schemas
        ):
            raise ValueError(
                "Execution ToolRegistry cannot register reserved action: "
                f"{UPDATE_PLAN_ACTION_NAME}"
            )
        self._action_schemas = [UPDATE_PLAN_SCHEMA, *execution_schemas]

    def run(self, task: str) -> AgentResult:
        context = ConversationContext(
            SYSTEM_PROMPT,
            task,
            max_chars=self._max_context_chars,
            max_messages=self._max_context_messages,
        )
        task_state = TaskState()
        progress_tracker = ProgressTracker()
        failures = _FailureTracker()
        current_step = 0

        try:
            for step in range(1, self._max_steps + 1):
                current_step = step
                try:
                    _notify_presentation(self._on_model_request_started)
                    context.set_runtime_state(task_state.snapshot_for_model())
                    task_state.clear_pending_completion_review()
                    prepared_messages = context.messages_for_model()
                    prepared_action_schemas = deepcopy(self._action_schemas)
                    response = self._request_model(
                        prepared_messages,
                        prepared_action_schemas,
                    )
                finally:
                    _notify_presentation(self._on_model_request_finished)

                if not response.tool_calls:
                    if (
                        step < self._max_steps
                        and task_state.request_completion_review()
                    ):
                        continue
                    return self._result(
                        task_state,
                        status="completed",
                        answer=response.text or "",
                        steps=step,
                    )

                assistant_message = self._assistant_tool_call_message(
                    response.text,
                    response.tool_calls,
                )
                progress_before = _progress_facts(task_state)
                progress_actions: list[ProgressAction] = []
                tool_messages: list[dict[str, Any]] = []
                for tool_call in response.tool_calls:
                    dispatch_started = perf_counter()
                    plan_before = task_state.plan
                    if tool_call.name == UPDATE_PLAN_ACTION_NAME:
                        result = apply_plan_action(task_state, tool_call.arguments)
                    else:
                        result = self._tools.dispatch(
                            tool_call.name,
                            tool_call.arguments,
                        )
                        if self._tools.confirmed_workspace_change(
                            tool_call.name,
                            result,
                        ):
                            task_state.record_workspace_change()
                        if tool_call.name == "run_command":
                            task_state.record_run_command(
                                model_step=step,
                                result=result,
                            )
                    progress_actions.append(
                        ProgressAction(
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            result=result,
                            workspace_revision=task_state.workspace_revision,
                            plan_before=plan_before,
                            plan_after=task_state.plan,
                        )
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
                    if (
                        tool_call.name != UPDATE_PLAN_ACTION_NAME
                        and self._on_tool_event is not None
                    ):
                        _notify_presentation(
                            self._on_tool_event,
                            ToolEvent(
                                tool_call=tool_call,
                                result=result,
                                dispatch_duration_ms=dispatch_duration_ms,
                                truncated=_tool_result_is_truncated(result),
                            )
                        )
                    if failures.record(tool_call, result):
                        return self._result(
                            task_state,
                            status="repeated_failure",
                            answer=None,
                            steps=step,
                            message=(
                                "Three consecutive identical action failures "
                                f"occurred for {tool_call.name}."
                            ),
                        )
                context.add_tool_cycle(assistant_message, tool_messages)
                progress_decision = progress_tracker.evaluate_turn(
                    task_state.progress,
                    before=progress_before,
                    after=_progress_facts(task_state),
                    actions=tuple(progress_actions),
                )
                if progress_decision == "terminate_no_progress":
                    return self._result(
                        task_state,
                        status="no_progress",
                        answer=None,
                        steps=step,
                        message=(
                            "No material progress was detected after one bounded "
                            "recovery attempt."
                        ),
                    )
                if (
                    progress_decision == "request_recovery"
                    and step >= self._max_steps
                ):
                    return self._result(
                        task_state,
                        status="max_steps",
                        answer=None,
                        steps=self._max_steps,
                        message=(
                            f"Maximum model decisions reached: {self._max_steps}."
                        ),
                    )

            return self._result(
                task_state,
                status="max_steps",
                answer=None,
                steps=self._max_steps,
                message=f"Maximum model decisions reached: {self._max_steps}.",
            )
        except KeyboardInterrupt:
            return self._result(
                task_state,
                status="user_interrupt",
                answer=None,
                steps=current_step,
                message="The run was interrupted by the user.",
            )
        except ModelAPIError as exc:
            return self._result(
                task_state,
                status="fatal_api_error",
                answer=None,
                steps=current_step,
                message=exc.safe_message,
            )
        except Exception:
            return self._result(
                task_state,
                status="runtime_error",
                answer=None,
                steps=current_step,
                message="An unexpected internal runtime error occurred.",
            )

    @staticmethod
    def _result(
        task_state: TaskState,
        *,
        status: TerminationReason,
        answer: str | None,
        steps: int,
        message: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            status=status,
            answer=answer,
            steps=steps,
            message=message,
            verification_status=task_state.verification_status,
            workspace_revision=task_state.workspace_revision,
            verified_revision=task_state.verified_revision,
            last_verification=task_state.verification.last_attempt,
            plan_status=task_state.plan_status,
            unfinished_steps=task_state.unfinished_steps,
            blocked_steps=task_state.blocked_steps,
        )

    def _request_model(
        self,
        messages: list[dict[str, Any]],
        action_schemas: list[dict[str, Any]],
    ) -> ModelResponse:
        for attempt in range(MODEL_REQUEST_ATTEMPTS):
            try:
                return self._client.complete(
                    deepcopy(messages),
                    deepcopy(action_schemas),
                )
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


def _progress_facts(task_state: TaskState) -> ProgressFacts:
    return ProgressFacts(
        workspace_revision=task_state.workspace_revision,
        verification_status=task_state.verification_status,
    )


def _notify_presentation(
    callback: Callable[..., None] | None,
    *args: object,
) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        pass

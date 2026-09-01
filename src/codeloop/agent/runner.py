"""The explicit decision-action-observation agent loop."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any, Literal, cast

from ..control import (
    ActionFingerprint,
    ApprovalPolicy,
    ApprovalSubject,
    AuthorizationScope,
    InteractionAction,
    InteractionProvider,
    InteractionRequest,
    InteractionResponse,
    TestScope,
)
from ..execution.command_policy import CommandDescription
from ..execution.tools import ToolRegistry, ToolResult
from ..model.client import ModelAPIError, ModelClient, ModelResponse, ToolCall
from .context import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_CONTEXT_MESSAGES,
    ConversationContext,
)
from .conversation import PublicConversationTurn
from .events import (
    CoreActionName,
    CoreActionEvent,
    CoreActionEventHandler,
    ModelRequestHandler,
    RecoveryEvent,
    RecoveryEventHandler,
    ReviewFindingProjection,
    ToolEvent,
    ToolEventHandler,
)
from .plan import (
    PlanStep,
    UPDATE_PLAN_ACTION_NAME,
    UPDATE_PLAN_SCHEMA,
    apply_plan_action,
)
from .human_interaction import (
    REQUEST_USER_INPUT_ACTION_NAME,
    REQUEST_USER_INPUT_SCHEMA,
    InteractionValidationError,
    parse_interaction_request,
)
from ..prompts import SYSTEM_PROMPT
from .progress import ProgressAction, ProgressFacts, ProgressTracker
from .repository import (
    UPDATE_WORKING_SET_ACTION_NAME,
    UPDATE_WORKING_SET_SCHEMA,
    apply_working_set_action,
)
from .review import (
    UPDATE_REVIEW_FINDINGS_ACTION_NAME,
    UPDATE_REVIEW_FINDINGS_SCHEMA,
    apply_review_findings_action,
)
from .requirements import (
    UPDATE_REQUIREMENTS_ACTION_NAME,
    UPDATE_REQUIREMENTS_SCHEMA,
    apply_requirements_action,
)
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
    "interaction_required",
]
MODEL_REQUEST_ATTEMPTS = 3
MODEL_RETRY_DELAYS = (0.5, 1.0)
DEFAULT_MAX_STEPS = 30
MAX_CONFIGURED_STEPS = 100
_CORE_ACTION_NAMES = frozenset(
    {
        UPDATE_PLAN_ACTION_NAME,
        UPDATE_REQUIREMENTS_ACTION_NAME,
        UPDATE_WORKING_SET_ACTION_NAME,
        UPDATE_REVIEW_FINDINGS_ACTION_NAME,
        REQUEST_USER_INPUT_ACTION_NAME,
    }
)
_INVALID_VISUAL_SEQUENCE_MESSAGE = (
    "read_image must be completed in a visual-only action turn. Read the visual "
    "sources first, then use the next multimodal decision to update requirements "
    "or continue the task."
)
_INVALID_INTERACTION_SEQUENCE_MESSAGE = (
    "request_user_input must be the only call in its model decision. Complete "
    "the Human Interaction first, then continue in the next decision."
)


@dataclass(frozen=True)
class _ApprovalOutcome:
    result: ToolResult | None
    public_note: str | None = None
    blocked: bool = False
    interaction_required: bool = False


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
        max_steps: int = DEFAULT_MAX_STEPS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_context_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES,
        on_tool_event: ToolEventHandler | None = None,
        on_core_action_event: CoreActionEventHandler | None = None,
        on_recovery_event: RecoveryEventHandler | None = None,
        interaction_provider: InteractionProvider | None = None,
        approval_policy: ApprovalPolicy | None = None,
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
        self._on_core_action_event = on_core_action_event
        self._on_recovery_event = on_recovery_event
        self._interaction_provider = interaction_provider
        self._approval_policy = approval_policy or ApprovalPolicy()
        self._on_model_request_started = on_model_request_started
        self._on_model_request_finished = on_model_request_finished
        execution_schemas = self._tools.schemas
        collisions = set(_CORE_ACTION_NAMES.intersection(self._tools.names))
        collisions.update(
            schema.get("function", {}).get("name")
            for schema in execution_schemas
            if schema.get("function", {}).get("name") in _CORE_ACTION_NAMES
        )
        if collisions:
            raise ValueError(
                "Execution ToolRegistry cannot register reserved action(s): "
                + ", ".join(sorted(collisions))
            )
        self._action_schemas = [
            UPDATE_PLAN_SCHEMA,
            UPDATE_REQUIREMENTS_SCHEMA,
            UPDATE_WORKING_SET_SCHEMA,
            UPDATE_REVIEW_FINDINGS_SCHEMA,
            REQUEST_USER_INPUT_SCHEMA,
            *execution_schemas,
        ]

    def run(
        self,
        task: str,
        *,
        previous_turns: Sequence[PublicConversationTurn] = (),
    ) -> AgentResult:
        authorization = AuthorizationScope()
        try:
            return self._run(
                task,
                previous_turns=previous_turns,
                authorization=authorization,
            )
        finally:
            authorization.clear()
            self._tools.discard_pending_visuals()

    def _run(
        self,
        task: str,
        *,
        previous_turns: Sequence[PublicConversationTurn] = (),
        authorization: AuthorizationScope,
    ) -> AgentResult:
        context = ConversationContext(
            SYSTEM_PROMPT,
            task,
            previous_turns=previous_turns,
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
                    pending_visuals = self._tools.snapshot_pending_visuals()
                    if pending_visuals:
                        prepared_messages.append(
                            _visual_request_message(pending_visuals)
                        )
                    del pending_visuals
                    response = self._request_model(
                        prepared_messages,
                        prepared_action_schemas,
                    )
                finally:
                    _notify_presentation(self._on_model_request_finished)

                has_read_image = any(
                    call.name == "read_image" for call in response.tool_calls
                )
                pure_visual_collection = bool(response.tool_calls) and all(
                    call.name == "read_image" for call in response.tool_calls
                )
                invalid_visual_sequence = (
                    has_read_image and not pure_visual_collection
                )
                has_user_interaction = any(
                    call.name == REQUEST_USER_INPUT_ACTION_NAME
                    for call in response.tool_calls
                )
                pure_user_interaction = (
                    len(response.tool_calls) == 1
                    and response.tool_calls[0].name
                    == REQUEST_USER_INPUT_ACTION_NAME
                )
                invalid_interaction_sequence = (
                    has_user_interaction and not pure_user_interaction
                )
                if not has_read_image and not invalid_interaction_sequence:
                    self._tools.consume_pending_visuals()

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
                human_decision_notes: list[str] = []
                repeated_failure_name: str | None = None
                interaction_required = False
                for tool_call in response.tool_calls:
                    dispatch_started = perf_counter()
                    plan_before = task_state.plan
                    approval_blocked = False
                    call_skipped = False
                    execution_dispatched = False
                    if invalid_interaction_sequence:
                        result = {
                            "ok": False,
                            "error_code": "invalid_action_sequence",
                            "message": _INVALID_INTERACTION_SEQUENCE_MESSAGE,
                        }
                    elif invalid_visual_sequence:
                        result = {
                            "ok": False,
                            "error_code": "invalid_action_sequence",
                            "message": _INVALID_VISUAL_SEQUENCE_MESSAGE,
                        }
                    elif interaction_required:
                        result = {
                            "ok": False,
                            "error_code": "interaction_required",
                            "message": (
                                "A prior action in this decision requires a "
                                "Human response before later calls can run."
                            ),
                        }
                        approval_blocked = True
                    elif repeated_failure_name is not None:
                        result = {
                            "ok": False,
                            "error_code": "repeated_failure",
                            "message": (
                                "This action was not executed because the "
                                "task reached the repeated-failure limit."
                            ),
                        }
                        call_skipped = True
                    elif tool_call.name == UPDATE_PLAN_ACTION_NAME:
                        result = apply_plan_action(task_state, tool_call.arguments)
                    elif tool_call.name == UPDATE_REQUIREMENTS_ACTION_NAME:
                        result = apply_requirements_action(
                            task_state,
                            tool_call.arguments,
                        )
                    elif tool_call.name == UPDATE_WORKING_SET_ACTION_NAME:
                        result = apply_working_set_action(
                            task_state,
                            tool_call.arguments,
                        )
                    elif tool_call.name == UPDATE_REVIEW_FINDINGS_ACTION_NAME:
                        result = apply_review_findings_action(
                            task_state,
                            tool_call.arguments,
                        )
                    elif tool_call.name == REQUEST_USER_INPUT_ACTION_NAME:
                        try:
                            request = parse_interaction_request(
                                tool_call.arguments
                            )
                        except InteractionValidationError as exc:
                            result = {
                                "ok": False,
                                "error_code": "invalid_arguments",
                                "message": exc.message,
                            }
                        else:
                            response_value = self._request_human_interaction(
                                request
                            )
                            if response_value.status == "interrupted":
                                return self._result(
                                    task_state,
                                    status="user_interrupt",
                                    answer=None,
                                    steps=step,
                                    message="The run was interrupted by the user.",
                                )
                            if response_value.status == "unavailable":
                                return self._result(
                                    task_state,
                                    status="interaction_required",
                                    answer=None,
                                    steps=step,
                                    message=(
                                        "The task requires Human Interaction, "
                                        "but no interactive response is available. "
                                        "Run CodeLoop in interactive mode to continue."
                                    ),
                                )
                            result = _human_interaction_result(
                                request,
                                response_value,
                            )
                            if (
                                request.kind in {"clarify", "choose"}
                                and response_value.answer
                            ):
                                authorization.record_human_response_basis(
                                    response_value.answer
                                )
                            _record_model_authorization(
                                request,
                                response_value,
                                authorization,
                            )
                            authorization.record_interaction(tool_call.id)
                    else:
                        preflight = self._tools.preflight_command(
                            tool_call.name,
                            tool_call.arguments,
                        )
                        approval = _ApprovalOutcome(result=None)
                        if preflight is not None and preflight.error is not None:
                            result = preflight.error
                        else:
                            if preflight is not None:
                                description = preflight.description
                                if description is None:
                                    raise RuntimeError(
                                        "Command preflight had no description"
                                    )
                                approval = self._authorize_execution(
                                    description,
                                    authorization,
                                    task=task,
                                    authorization_basis=(
                                        _command_authorization_basis(
                                            tool_call.arguments
                                        )
                                    ),
                                )
                            if approval.result is None:
                                result = self._tools.dispatch(
                                    tool_call.name,
                                    tool_call.arguments,
                                )
                                execution_dispatched = True
                            else:
                                result = approval.result
                                approval_blocked = approval.blocked
                                interaction_required = (
                                    interaction_required
                                    or approval.interaction_required
                                )
                            if approval.public_note:
                                human_decision_notes.append(approval.public_note)
                        task_state.record_execution_evidence(
                            tool_name=tool_call.name,
                            result=result,
                        )
                        if self._tools.confirmed_workspace_change(
                            tool_call.name,
                            result,
                        ):
                            data = result.get("data")
                            changed_path = (
                                data.get("path")
                                if (
                                    tool_call.name in {"edit_file", "write_file"}
                                    and isinstance(data, dict)
                                )
                                else None
                            )
                            task_state.record_workspace_change(
                                changed_path if isinstance(changed_path, str) else None
                            )
                        if tool_call.name == "run_command" and execution_dispatched:
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
                        not invalid_visual_sequence
                        and not invalid_interaction_sequence
                        and not approval_blocked
                        and not call_skipped
                        and tool_call.name not in _CORE_ACTION_NAMES
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
                    elif (
                        not invalid_visual_sequence
                        and not invalid_interaction_sequence
                        and not approval_blocked
                        and not call_skipped
                        and tool_call.name in _CORE_ACTION_NAMES
                        and tool_call.name != REQUEST_USER_INPUT_ACTION_NAME
                        and self._on_core_action_event is not None
                    ):
                        _notify_presentation(
                            self._on_core_action_event,
                            _core_action_event(tool_call, result, task_state),
                        )
                    if (
                        not approval_blocked
                        and not call_skipped
                        and failures.record(tool_call, result)
                        and repeated_failure_name is None
                    ):
                        repeated_failure_name = tool_call.name
                if pure_user_interaction:
                    context.add_interaction_cycle(
                        assistant_message,
                        tool_messages[0],
                    )
                else:
                    context.add_tool_cycle(assistant_message, tool_messages)
                if human_decision_notes:
                    context.add_public_note(
                        "Human authorization note: "
                        + " | ".join(human_decision_notes)
                    )
                if interaction_required:
                    return self._result(
                        task_state,
                        status="interaction_required",
                        answer=None,
                        steps=step,
                        message=(
                            "The task requires Human approval before the "
                            "command can be executed, but no interactive "
                            "response is available. Run CodeLoop in interactive "
                            "mode to continue."
                        ),
                    )
                if repeated_failure_name is not None:
                    return self._result(
                        task_state,
                        status="repeated_failure",
                        answer=None,
                        steps=step,
                        message=(
                            "Three consecutive identical action failures "
                            f"occurred for {repeated_failure_name}."
                        ),
                    )
                progress_decision = progress_tracker.evaluate_turn(
                    task_state.progress,
                    before=progress_before,
                    after=_progress_facts(task_state),
                    actions=tuple(progress_actions),
                )
                if progress_decision == "request_recovery":
                    _notify_presentation(
                        self._on_recovery_event,
                        RecoveryEvent(),
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

    def _request_human_interaction(
        self,
        request: InteractionRequest,
    ) -> InteractionResponse:
        provider = self._interaction_provider
        if provider is None:
            if request.kind == "inform":
                return InteractionResponse(status="answered", answer="acknowledged")
            return InteractionResponse(status="unavailable")
        try:
            return provider.interact(request)
        except KeyboardInterrupt:
            return InteractionResponse(status="interrupted")
        except EOFError:
            return InteractionResponse(status="unavailable")
        except Exception:
            return InteractionResponse(status="unavailable")

    def _authorize_execution(
        self,
        description: CommandDescription,
        authorization: AuthorizationScope,
        *,
        task: str,
        authorization_basis: str | None,
    ) -> _ApprovalOutcome:
        fingerprint = ActionFingerprint.command(
            description.command,
            description.cwd,
        )
        test_scope = _test_scope_for_command(description, fingerprint)
        traceable_basis = (
            authorization_basis
            if authorization_basis is not None
            and authorization.basis_is_traceable(
                authorization_basis,
                task,
            )
            else None
        )
        if description.category in {"test", "program_execution"}:
            if (
                traceable_basis is not None
                and not _has_related_reusable_scope(
                    authorization,
                    description.category,
                )
            ):
                authorization.approve_reusable(
                    description.category,
                    fingerprint,
                    test_scope=test_scope,
                    command=description.display_command,
                    cwd=description.cwd,
                )
            elif not authorization.authorizes(
                description.category,
                fingerprint,
                test_scope=test_scope,
            ):
                # A request outside the existing semantic TEST scope, or a
                # changed exact PROGRAM_EXECUTION action, cannot silently reuse
                # the earlier natural-language basis.
                traceable_basis = None
        elif traceable_basis is not None:
            if fingerprint in authorization.used_one_shot_bases:
                traceable_basis = None
            else:
                authorization.mark_one_shot_basis_used(fingerprint)
        subject = ApprovalSubject(
            description="Execute the validated command",
            category=description.category,
            command=description.command,
            cwd=description.cwd,
            fingerprint=fingerprint,
            reason=description.reason,
            test_scope=test_scope,
            authorization_basis=traceable_basis,
        )
        decision = self._approval_policy.decide(subject, authorization)
        data = {
            "category": description.category,
            "cwd": description.cwd,
        }
        if decision == "allow":
            return _ApprovalOutcome(result=None)
        if decision == "deny":
            if fingerprint in authorization.denied_fingerprints:
                return _ApprovalOutcome(
                    result={
                        "ok": False,
                        "error_code": "user_denied",
                        "message": (
                            "The command was not executed because the same "
                            "action was already denied in this task."
                        ),
                        "data": data,
                    },
                    blocked=True,
                )
            return _ApprovalOutcome(
                result={
                    "ok": False,
                    "error_code": "permission_denied",
                    "message": "The command is denied by Runtime safety policy.",
                    "data": data,
                },
                blocked=True,
            )
        if decision == "inform":
            authorization.consume_one_shot(fingerprint)
            response = self._request_human_interaction(
                InteractionRequest(
                    kind="inform",
                    prompt=description.reason,
                    action=_interaction_action_for_command(description),
                )
            )
            if response.status == "interrupted":
                raise KeyboardInterrupt
            # INFORM is notification-only. Unavailable presentation does not
            # revoke the already-traceable authorization and never creates it.
            return _ApprovalOutcome(
                result=None,
                public_note=(
                    f"notified authorized {description.category} action"
                    if response.status == "answered"
                    else None
                ),
            )

        interaction_kind = (
            "re_approve"
            if _has_related_reusable_scope(authorization, description.category)
            else "approve"
        )
        previous_command, previous_cwd, scope_change = _scope_change_context(
            authorization,
            description,
            test_scope,
        )
        response = self._request_human_interaction(
            InteractionRequest(
                kind=interaction_kind,
                prompt=description.reason,
                action=_interaction_action_for_command(
                    description,
                    previous_command=previous_command,
                    previous_cwd=previous_cwd,
                    scope_change=scope_change,
                ),
            )
        )
        if response.status == "interrupted":
            raise KeyboardInterrupt
        if response.status != "answered" or response.approved is None:
            return _ApprovalOutcome(
                result={
                    "ok": False,
                    "error_code": "approval_unavailable",
                    "message": (
                        "The command was not executed because Human approval "
                        "was unavailable."
                    ),
                    "data": data,
                },
                blocked=True,
                interaction_required=True,
            )
        if response.approved is not True:
            authorization.deny(fingerprint)
            return _ApprovalOutcome(
                result={
                    "ok": False,
                    "error_code": "user_denied",
                    "message": (
                        "The command was not executed because the user denied it."
                    ),
                    "data": data,
                },
                public_note=f"denied {description.category} action",
                blocked=True,
            )
        authorization.approve_reusable(
            description.category,
            fingerprint,
            test_scope=test_scope,
            command=description.display_command,
            cwd=description.cwd,
        )
        return _ApprovalOutcome(
            result=None,
            public_note=f"approved {description.category} action",
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


def _core_action_event(
    tool_call: ToolCall,
    result: ToolResult,
    task_state: TaskState,
) -> CoreActionEvent:
    """Project only the successful Action's own bounded state for presentation."""
    if result.get("ok") is not True:
        return CoreActionEvent(
            name=cast(CoreActionName, tool_call.name),
            call_id=tool_call.id,
            result=result,
        )
    if tool_call.name == UPDATE_REQUIREMENTS_ACTION_NAME:
        requirements = task_state.requirements.requirements
        sources = tuple(
            dict.fromkeys(
                source
                for requirement in requirements
                for source in (requirement.source.path or requirement.source.url,)
                if source is not None
            )
        )
        return CoreActionEvent(
            name=UPDATE_REQUIREMENTS_ACTION_NAME,
            call_id=tool_call.id,
            result=result,
            requirement_count=len(requirements),
            requirement_sources=sources,
        )
    if tool_call.name == UPDATE_PLAN_ACTION_NAME:
        return CoreActionEvent(
            name=UPDATE_PLAN_ACTION_NAME,
            call_id=tool_call.id,
            result=result,
            plan_steps=(
                task_state.plan.steps if task_state.plan is not None else ()
            ),
        )
    if tool_call.name == UPDATE_REVIEW_FINDINGS_ACTION_NAME:
        return CoreActionEvent(
            name=UPDATE_REVIEW_FINDINGS_ACTION_NAME,
            call_id=tool_call.id,
            result=result,
            review_findings=tuple(
                ReviewFindingProjection(
                    title=finding.title,
                    finding_type=finding.finding_type,
                    priority=finding.priority,
                )
                for finding in task_state.review_state.findings
            ),
        )
    if tool_call.name == REQUEST_USER_INPUT_ACTION_NAME:
        return CoreActionEvent(
            name=REQUEST_USER_INPUT_ACTION_NAME,
            call_id=tool_call.id,
            result=result,
        )
    return CoreActionEvent(
        name=UPDATE_WORKING_SET_ACTION_NAME,
        call_id=tool_call.id,
        result={"ok": True},
    )


def _visual_request_message(pending_visuals: Sequence[Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for payload in pending_visuals:
        descriptor = payload.descriptor
        encoded = base64.b64encode(payload.raw_bytes).decode("ascii")
        content.extend(
            (
                {
                    "type": "text",
                    "text": f"Visual source: {descriptor.source_label}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{descriptor.mime_type};base64,{encoded}"
                    },
                },
            )
        )
    return {"role": "user", "content": content}


def _interaction_action_for_command(
    description: CommandDescription,
    *,
    previous_command: tuple[str, ...] = (),
    previous_cwd: str | None = None,
    scope_change: str | None = None,
) -> InteractionAction:
    return InteractionAction(
        description="执行已校验的本地命令",
        category=description.category,
        command=description.display_command,
        cwd=description.cwd,
        workspace_root=description.workspace_root,
        previous_command=previous_command,
        previous_cwd=previous_cwd,
        scope_change=scope_change,
    )


def _test_scope_for_command(
    description: CommandDescription,
    fingerprint: ActionFingerprint,
) -> TestScope | None:
    if description.category != "test":
        return None
    facts = description.test_scope
    if facts is None:
        return TestScope(
            fingerprint=fingerprint,
            cwd=description.cwd,
            command=description.display_command,
        )
    return TestScope(
        fingerprint=fingerprint,
        family=facts.family,
        cwd=description.cwd,
        all_tests=facts.all_tests,
        targets=frozenset(facts.targets),
        command=description.display_command,
    )


def _scope_change_context(
    authorization: AuthorizationScope,
    description: CommandDescription,
    requested_test_scope: TestScope | None,
) -> tuple[tuple[str, ...], str | None, str | None]:
    if description.category == "test" and requested_test_scope is not None:
        previous = authorization.related_test_scope(requested_test_scope)
        if previous is None:
            return (), None, None
        return (
            previous.command,
            previous.cwd,
            _test_scope_change_description(previous, requested_test_scope),
        )
    if description.category == "program_execution":
        previous = authorization.related_program_action()
        if previous is not None:
            command, cwd = previous
            return (
                command,
                cwd,
                "本地程序命令或工作目录发生变化，原授权只覆盖之前的精确动作。",
            )
    return (), None, None


def _test_scope_change_description(
    previous: TestScope,
    requested: TestScope,
) -> str:
    if previous.cwd != requested.cwd:
        return "测试工作目录发生变化，原授权不覆盖新的项目内目录。"
    if previous.family != requested.family:
        return "测试运行器或测试体系发生变化，原授权不覆盖当前测试命令。"
    if requested.all_tests and not previous.all_tests:
        family = {
            "node": "Node",
            "pytest": "pytest",
            "npm:test": "npm",
            "pnpm:test": "pnpm",
            "yarn:test": "yarn",
        }.get(requested.family or "", "当前")
        return f"从指定测试扩大为当前工作目录下的全部 {family} 测试。"
    if requested.targets and not requested.targets.issubset(previous.targets):
        return "当前请求增加了之前未批准的测试目标。"
    return "当前测试命令不在之前批准的确定性测试范围内。"


def _command_authorization_basis(arguments_json: str) -> str | None:
    """Read a preflight-validated declaration without interpreting intent."""
    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arguments, dict):
        return None
    value = arguments.get("authorization_basis")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _has_related_reusable_scope(
    authorization: AuthorizationScope,
    category: str,
) -> bool:
    if category == "test":
        return bool(authorization.test_scopes)
    if category == "program_execution":
        return bool(authorization.program_actions)
    return False


def _human_interaction_result(
    request: InteractionRequest,
    response: InteractionResponse,
) -> ToolResult:
    data: dict[str, Any] = {
        "response_type": "human_interaction_response",
        "kind": request.kind,
    }
    if response.answer is not None:
        data["answer"] = response.answer[:2_000]
    if response.selected_option_id is not None:
        data["selected_option_id"] = response.selected_option_id
    if response.approved is not None:
        data["approved"] = response.approved
    return {"ok": True, "data": data}


def _record_model_authorization(
    request: InteractionRequest,
    response: InteractionResponse,
    authorization: AuthorizationScope,
) -> None:
    if (
        request.kind not in {"approve", "re_approve"}
        or response.approved is not True
        or request.action is None
        or not request.action.command
        or request.action.cwd is None
    ):
        return
    fingerprint = ActionFingerprint.command(
        request.action.command,
        request.action.cwd,
    )
    if request.action.category in {"test", "program_execution"}:
        authorization.approve_reusable(
            request.action.category,
            fingerprint,
            test_scope=(
                TestScope(
                    fingerprint=fingerprint,
                    cwd=request.action.cwd,
                    command=request.action.command,
                )
                if request.action.category == "test"
                else None
            ),
            command=request.action.command,
            cwd=request.action.cwd,
        )
    else:
        authorization.approve_one_shot(fingerprint)


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

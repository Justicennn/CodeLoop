"""Stage 7D conservative progress and stall-recovery tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent.runner as runner_module
from codeloop.agent.context import ConversationContext
from codeloop.agent.events import RecoveryEvent
from codeloop.agent.plan import PlanStep, TaskPlan, UPDATE_PLAN_ACTION_NAME
from codeloop.agent.progress import (
    RECENT_DIGEST_LIMIT,
    ProgressAction,
    ProgressFacts,
    ProgressState,
    ProgressTracker,
    action_digest,
    observation_digest,
)
from codeloop.agent.runner import AgentRunner
from codeloop.agent.task_state import TaskState
from codeloop.execution.tools import ToolDefinition, ToolRegistry
from codeloop.execution.workspace import Workspace
from codeloop.interaction.cli import EXIT_CODES
from codeloop.model.client import ModelAPIError, ModelResponse, ToolCall


class FakeClient:
    def __init__(self, responses: list[ModelResponse | BaseException]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "tools": tools})
        response = next(self._responses)
        if isinstance(response, BaseException):
            raise response
        return response


def _facts(revision: int = 0, status: str = "not_required") -> ProgressFacts:
    return ProgressFacts(
        workspace_revision=revision,
        verification_status=status,  # type: ignore[arg-type]
    )


def _action(
    name: str,
    arguments: dict[str, Any] | str,
    result: dict[str, Any],
    *,
    revision: int = 0,
    plan_before: TaskPlan | None = None,
    plan_after: TaskPlan | None = None,
) -> ProgressAction:
    raw_arguments = (
        arguments if isinstance(arguments, str) else json.dumps(arguments)
    )
    return ProgressAction(
        name=name,
        arguments=raw_arguments,
        result=result,
        workspace_revision=revision,
        plan_before=plan_before,
        plan_after=plan_after,
    )


def _read_action(content: str = "1: value", *, revision: int = 0) -> ProgressAction:
    return _action(
        "read_file",
        {"path": "value.txt"},
        {
            "ok": True,
            "data": {
                "path": "value.txt",
                "start_line": 1,
                "end_line": 1,
                "total_lines": 1,
                "content": content,
                "truncated": False,
            },
        },
        revision=revision,
    )


def _evaluate(
    tracker: ProgressTracker,
    state: ProgressState,
    action: ProgressAction,
    *,
    before: ProgressFacts | None = None,
    after: ProgressFacts | None = None,
) -> str:
    return tracker.evaluate_turn(
        state,
        before=before or _facts(),
        after=after or _facts(),
        actions=(action,),
    )


def _plan(active: str, *, terminal: str | None = None) -> TaskPlan:
    statuses = {
        "a": "completed" if terminal == "a" else (
            "in_progress" if active == "a" else "pending"
        ),
        "b": "blocked" if terminal == "b" else (
            "in_progress" if active == "b" else "pending"
        ),
    }
    return TaskPlan(
        revision=1,
        steps=tuple(
            PlanStep(
                id=step_id,
                description=f"Step {step_id}",
                status=status,  # type: ignore[arg-type]
                blocked_reason=("blocked" if status == "blocked" else None),
            )
            for step_id, status in statuses.items()
        ),
    )


def _plan_action(
    before: TaskPlan,
    after: TaskPlan,
    mode: str = "update",
) -> ProgressAction:
    return _action(
        UPDATE_PLAN_ACTION_NAME,
        {"mode": mode, "steps": []},
        {
            "ok": True,
            "data": {
                "changed": True,
                "mode": mode,
                "revision": after.revision,
                "plan_status": after.status,
                "changed_step_ids": [step.id for step in after.steps],
            },
        },
        plan_before=before,
        plan_after=after,
    )


def test_progress_state_is_serializable_and_snapshot_is_minimal() -> None:
    state = ProgressState(
        status="possible_stall",
        no_progress_turns=4,
        repeating_pattern_turns=2,
        last_no_progress_pattern="pattern",
        recovery_active=True,
        recovery_reason="extended_no_progress",
        recent_observation_digests=("observation",),
        recent_active_transition_digests=("transition",),
    )

    json.dumps(asdict(state))
    snapshot = state.snapshot_for_model()

    assert snapshot is not None
    assert set(snapshot) == {"status", "reason", "instruction"}
    serialized = json.dumps(snapshot)
    for forbidden in (
        "no_progress_turns",
        "repeating_pattern_turns",
        "fingerprint",
        "digest",
        "pattern",
    ):
        assert forbidden not in serialized


def test_action_and_observation_fingerprints_ignore_non_material_variation() -> None:
    assert action_digest("search_code", '{"query":"x","path":"."}') == (
        action_digest("search_code", '{"path":".","query":"x"}')
    )

    first = _action(
        "search_code",
        {"query": "first"},
        {
            "ok": True,
            "data": {
                "query": "first",
                "matches": [],
                "count": 0,
                "files": 0,
                "truncated": False,
                "engine": "rg",
            },
        },
    )
    second = _action(
        "search_code",
        {"query": "second"},
        {
            "ok": True,
            "data": {
                "query": "second",
                "matches": [],
                "count": 0,
                "files": 0,
                "truncated": False,
                "engine": "python",
            },
        },
    )
    run_one = _action(
        "run_command",
        {"command": ["check-one"]},
        {
            "ok": False,
            "error_code": "command_failed",
            "message": "first wording",
            "data": {
                "command": ["check-one"],
                "cwd": ".",
                "exit_code": 1,
                "stdout": "",
                "stderr": "failed",
                "timed_out": False,
                "duration_ms": 1,
            },
        },
    )
    run_two = _action(
        "run_command",
        {"command": ["check-two"]},
        {
            "ok": False,
            "error_code": "command_failed",
            "message": "second wording",
            "data": {
                "command": ["check-two"],
                "cwd": "subdir",
                "exit_code": 1,
                "stdout": "",
                "stderr": "failed",
                "timed_out": False,
                "duration_ms": 999,
            },
        },
    )

    assert observation_digest(first) == observation_digest(second)
    assert observation_digest(run_one) == observation_digest(run_two)


def test_repeating_observation_requests_recovery_then_terminates() -> None:
    tracker = ProgressTracker()
    state = ProgressState()
    action = _read_action()

    assert _evaluate(tracker, state, action) == "continue"
    assert [_evaluate(tracker, state, action) for _ in range(3)] == [
        "continue",
        "continue",
        "request_recovery",
    ]
    assert state.recovery_active is True
    assert state.snapshot_for_model() is not None
    assert [_evaluate(tracker, state, action) for _ in range(3)] == [
        "continue",
        "continue",
        "terminate_no_progress",
    ]


def test_different_actions_with_same_empty_observation_use_extended_limit() -> None:
    tracker = ProgressTracker()
    state = ProgressState()
    base_result = {
        "ok": True,
        "data": {
            "query": "ignored",
            "matches": [],
            "count": 0,
            "files": 0,
            "truncated": False,
            "engine": "rg",
        },
    }

    assert _evaluate(
        tracker,
        state,
        _action("search_code", {"query": "initial"}, base_result),
    ) == "continue"
    outcomes = [
        _evaluate(
            tracker,
            state,
            _action("search_code", {"query": f"query-{index}"}, base_result),
        )
        for index in range(5)
    ]

    assert outcomes[-1] == "request_recovery"
    assert state.recovery_reason == "extended_no_progress"


def test_attempt_metadata_is_not_independent_progress_but_new_output_is() -> None:
    tracker = ProgressTracker()
    state = ProgressState()
    failed = _action(
        "run_command",
        {"command": ["first"]},
        {
            "ok": False,
            "error_code": "command_failed",
            "data": {
                "exit_code": 1,
                "stdout": "",
                "stderr": "same evidence",
                "timed_out": False,
            },
        },
    )
    same_material = _action(
        "run_command",
        {"command": ["different"]},
        failed.result,
    )
    new_material = _action(
        "run_command",
        {"command": ["third"]},
        {
            "ok": False,
            "error_code": "command_timeout",
            "data": {
                "exit_code": None,
                "stdout": "new evidence",
                "stderr": "",
                "timed_out": True,
            },
        },
    )

    assert _evaluate(tracker, state, failed) == "continue"
    assert _evaluate(tracker, state, same_material) == "continue"
    assert state.no_progress_turns == 1
    assert _evaluate(tracker, state, new_material) == "continue"
    assert state.no_progress_turns == 0


def test_verification_status_change_is_progress_without_attempt_metadata() -> None:
    tracker = ProgressTracker()
    state = ProgressState(
        status="possible_stall",
        recovery_active=True,
        recovery_reason="extended_no_progress",
    )
    repeated = _read_action()
    state.recent_observation_digests = (observation_digest(repeated),)

    assert _evaluate(
        tracker,
        state,
        repeated,
        before=_facts(1, "unverified"),
        after=_facts(1, "verified"),
    ) == "continue"
    assert state.recovery_active is False


def test_plan_terminal_progress_and_bounded_active_switches() -> None:
    tracker = ProgressTracker()
    state = ProgressState()
    plan_a = _plan("a")
    plan_b = _plan("b")

    assert _evaluate(tracker, state, _plan_action(plan_a, plan_b)) == "continue"
    assert _evaluate(tracker, state, _plan_action(plan_b, plan_a)) == "continue"
    assert len(state.recent_active_transition_digests) == 2

    outcomes = []
    before, after = plan_a, plan_b
    for _ in range(3):
        outcomes.append(_evaluate(tracker, state, _plan_action(before, after)))
        before, after = after, before
    assert outcomes == ["continue", "continue", "request_recovery"]

    terminal_state = ProgressState(recovery_active=True, status="possible_stall")
    terminal_state.recovery_reason = "extended_no_progress"
    completed = _plan("", terminal="a")
    assert _evaluate(
        tracker,
        terminal_state,
        _plan_action(plan_a, completed),
    ) == "continue"
    assert terminal_state.recovery_active is False


def test_create_replan_and_noop_do_not_clear_recovery() -> None:
    tracker = ProgressTracker()
    state = ProgressState(
        status="possible_stall",
        recovery_active=True,
        recovery_reason="extended_no_progress",
    )
    plan_a = _plan("a")

    for mode in ("create", "replan", "update"):
        result = _evaluate(
            tracker,
            state,
            _plan_action(plan_a, plan_a, mode=mode),
        )
        assert result == "continue"
        assert state.recovery_active is True


def test_real_progress_keeps_recent_digest_memory_and_capacity() -> None:
    tracker = ProgressTracker()
    state = ProgressState()
    first = _read_action("first")
    first_digest = observation_digest(first)
    _evaluate(tracker, state, first)

    mutation = _action(
        "write_file",
        {"path": "new.txt", "content": "x"},
        {
            "ok": True,
            "data": {"path": "new.txt", "workspace_changed": True},
        },
        revision=1,
    )
    _evaluate(
        tracker,
        state,
        mutation,
        before=_facts(0),
        after=_facts(1, "unverified"),
    )
    assert first_digest in state.recent_observation_digests

    for index in range(RECENT_DIGEST_LIMIT + 1):
        _evaluate(tracker, state, _read_action(f"content-{index}"))
    assert len(state.recent_observation_digests) == RECENT_DIGEST_LIMIT
    assert first_digest not in state.recent_observation_digests


def test_progress_snapshot_replaces_and_survives_cycle_trimming() -> None:
    context = ConversationContext(
        "system",
        "task",
        max_chars=1_000,
        max_messages=20,
    )
    task_state = TaskState()
    task_state.progress.status = "possible_stall"
    task_state.progress.recovery_active = True
    task_state.progress.recovery_reason = "repeating_pattern"
    task_state.progress.no_progress_turns = 99
    task_state.progress.recent_observation_digests = ("private-digest",)
    context.set_runtime_state(task_state.snapshot_for_model())

    for identifier in ("old", "latest"):
        context.add_tool_cycle(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": identifier,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            [
                {
                    "role": "tool",
                    "tool_call_id": identifier,
                    "name": "read_file",
                    "content": "x" * 450,
                }
            ],
        )

    messages = context.messages_for_model()
    system = messages[0]["content"]
    assert '"progress"' in system
    assert "private-digest" not in system
    assert "no_progress_turns" not in system
    assert not any(message.get("tool_call_id") == "old" for message in messages)
    assert any(message.get("tool_call_id") == "latest" for message in messages)

    task_state.progress.status = "active"
    task_state.progress.recovery_active = False
    task_state.progress.recovery_reason = None
    context.set_runtime_state(task_state.snapshot_for_model())
    assert '"progress"' not in context.messages_for_model()[0]["content"]


def _read_response(identifier: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=[
            ToolCall(
                id=identifier,
                name="read_file",
                arguments='{"path":"value.txt"}',
            )
        ]
    )


def test_runner_injects_recovery_then_terminates_no_progress(
    tmp_path: Path,
) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    client = FakeClient([_read_response(f"read-{index}") for index in range(7)])

    recovery_events: list[RecoveryEvent] = []
    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        max_steps=7,
        on_recovery_event=recovery_events.append,
    ).run("Keep reading the same file.")

    assert result.status == "no_progress"
    assert result.steps == 7
    assert result.workspace_revision == 0
    assert result.verification_status == "not_required"
    assert len(recovery_events) == 1
    assert recovery_events[0].reason == "no_progress"
    recovery_snapshot = client.calls[4]["messages"][0]["content"]
    assert '"progress"' in recovery_snapshot
    assert '"status":"possible_stall"' in recovery_snapshot
    for forbidden in ("no_progress_turns", "fingerprint", "digest"):
        assert forbidden not in recovery_snapshot
    assert EXIT_CODES["no_progress"] == 1


def test_first_stall_on_last_step_remains_max_steps(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    client = FakeClient([_read_response(f"read-{index}") for index in range(4)])

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        max_steps=4,
    ).run("Keep reading.")

    assert result.status == "max_steps"
    assert result.steps == 4


def test_repeated_failure_keeps_priority_over_progress(tmp_path: Path) -> None:
    failed_reads = [
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id=f"missing-{index}",
                    name="read_file",
                    arguments='{"path":"missing.txt"}',
                )
            ]
        )
        for index in range(3)
    ]

    result = AgentRunner(
        FakeClient(failed_reads),
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Read a missing file repeatedly.")

    assert result.status == "repeated_failure"
    assert result.steps == 3


def test_no_progress_result_preserves_workspace_and_verification_facts(
    tmp_path: Path,
) -> None:
    write = ModelResponse(
        tool_calls=[
            ToolCall(
                id="write",
                name="write_file",
                arguments='{"path":"value.txt","content":"value"}',
            )
        ]
    )
    client = FakeClient(
        [write, *[_read_response(f"read-{index}") for index in range(7)]]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        max_steps=8,
    ).run("Create a file, then make no further progress.")

    assert result.status == "no_progress"
    assert result.workspace_revision == 1
    assert result.verification_status == "unverified"
    assert result.verified_revision is None


def test_progress_snapshot_is_stable_across_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    temporary = ModelAPIError(
        "temporary_api_error",
        "temporary",
        classification="retryable",
    )
    client = FakeClient(
        [
            *[_read_response(f"read-{index}") for index in range(4)],
            temporary,
            ModelResponse(text="Changed strategy."),
        ]
    )
    monkeypatch.setattr(runner_module, "sleep", lambda seconds: None)

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Recover from repetition.")

    assert result.status == "completed"
    assert client.calls[4] == client.calls[5]
    assert '"progress"' in client.calls[4]["messages"][0]["content"]


def test_complex_task_completes_without_false_stall(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    run_definition = registry._tools["run_command"]
    run_results = iter(
        [
            {
                "ok": False,
                "error_code": "command_failed",
                "message": "failed",
                "data": {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "test failed",
                    "timed_out": False,
                },
            },
            {
                "ok": True,
                "data": {
                    "exit_code": 0,
                    "stdout": "tests passed",
                    "stderr": "",
                    "timed_out": False,
                },
            },
        ]
    )
    registry._tools["run_command"] = ToolDefinition(
        run_definition.schema,
        lambda arguments: next(run_results),
    )

    initial_steps = [
        {"id": "build", "description": "Build", "status": "in_progress"},
        {"id": "verify", "description": "Verify", "status": "pending"},
    ]
    replanned_steps = [
        *initial_steps,
        {"id": "repair", "description": "Repair", "status": "pending"},
    ]
    completed_steps = [
        {**step, "status": "completed"} for step in replanned_steps
    ]

    def call(identifier: str, name: str, arguments: dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            tool_calls=[
                ToolCall(
                    id=identifier,
                    name=name,
                    arguments=json.dumps(arguments),
                )
            ]
        )

    client = FakeClient(
        [
            call(
                "plan",
                UPDATE_PLAN_ACTION_NAME,
                {"mode": "create", "steps": initial_steps},
            ),
            call("mkdir", "make_directory", {"path": "src"}),
            call("write", "write_file", {"path": "src/app.py", "content": "bad"}),
            call("run-fail", "run_command", {"command": ["check"]}),
            call(
                "replan",
                UPDATE_PLAN_ACTION_NAME,
                {
                    "mode": "replan",
                    "steps": replanned_steps,
                    "explanation": "The first verification exposed a defect.",
                },
            ),
            call(
                "edit",
                "edit_file",
                {"path": "src/app.py", "old_text": "bad", "new_text": "good"},
            ),
            call("run-pass", "run_command", {"command": ["check"]}),
            call(
                "complete-plan",
                UPDATE_PLAN_ACTION_NAME,
                {"mode": "update", "steps": completed_steps},
            ),
            ModelResponse(text="Built and verified."),
        ]
    )

    result = AgentRunner(client, tools=registry).run("Build a small project.")

    assert result.status == "completed"
    assert result.workspace_revision == 3
    assert result.verification_status == "verified"
    assert result.verified_revision == 3
    assert result.plan_status == "completed"


def test_read_only_review_completes_without_requiring_verification(
    tmp_path: Path,
) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="list",
                        name="list_files",
                        arguments='{"path":"."}',
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="search",
                        name="search_code",
                        arguments='{"query":"value","path":"."}',
                    )
                ]
            ),
            _read_response("read"),
            ModelResponse(text="Review complete."),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Review the project without changing it.")

    assert result.status == "completed"
    assert result.workspace_revision == 0
    assert result.verification_status == "not_required"

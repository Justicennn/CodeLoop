"""Stage 7C verification state and completion-contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent.runner as runner_module
from codeloop.agent.context import ConversationContext
from codeloop.agent.plan import UPDATE_PLAN_ACTION_NAME
from codeloop.agent.prompt import SYSTEM_PROMPT
from codeloop.agent.runner import AgentRunner
from codeloop.agent.task_state import TaskState
from codeloop.agent.verification import VerificationState
from codeloop.execution.tools import ToolDefinition, ToolRegistry
from codeloop.execution.workspace import Workspace
from codeloop.model.client import ModelAPIError, ModelResponse, ToolCall


SUCCESS = {
    "ok": True,
    "data": {"exit_code": 0, "timed_out": False, "stdout": "secret output"},
}
FAILURE = {
    "ok": False,
    "error_code": "command_failed",
    "message": "failed",
    "data": {"exit_code": 1, "timed_out": False, "stderr": "details"},
}


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


def _tool_call(
    identifier: str,
    name: str,
    arguments: dict[str, Any] | str,
) -> ToolCall:
    serialized = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=identifier, name=name, arguments=serialized)


def _registry_with_run_results(
    tmp_path: Path,
    results: list[dict[str, Any]],
) -> ToolRegistry:
    registry = ToolRegistry(Workspace(tmp_path))
    definition = registry._tools["run_command"]
    remaining = iter(results)
    registry._tools["run_command"] = ToolDefinition(
        definition.schema,
        lambda arguments: next(remaining),
    )
    return registry


def test_latest_attempt_for_current_revision_controls_status() -> None:
    state = TaskState()

    assert state.verification_status == "not_required"
    state.record_workspace_change()
    assert state.verification_status == "unverified"

    state.record_run_command(model_step=1, result=SUCCESS)
    first_success = state.verification.last_success
    assert state.verification_status == "verified"
    assert state.verified_revision == 1
    assert first_success is not None
    assert first_success.workspace_revision == 1

    state.record_run_command(model_step=2, result=FAILURE)
    assert state.verification_status == "unverified"
    assert state.verified_revision is None
    assert state.verification.last_success == first_success

    state.record_run_command(model_step=3, result=SUCCESS)
    assert state.verification_status == "verified"
    assert state.verified_revision == 1
    assert state.verification.last_attempt is not None
    assert state.verification.last_attempt.model_step == 3

    state.record_workspace_change()
    assert state.workspace_revision == 2
    assert state.verification_status == "unverified"
    assert state.verified_revision is None
    assert state.verification.last_success is not None
    assert state.verification.last_success.workspace_revision == 1


@pytest.mark.parametrize(
    "result,error_code,exit_code,timed_out",
    [
        (FAILURE, "command_failed", 1, False),
        (
            {
                "ok": False,
                "error_code": "command_timeout",
                "message": "timeout",
                "data": {"exit_code": None, "timed_out": True},
            },
            "command_timeout",
            None,
            True,
        ),
        (
            {
                "ok": False,
                "error_code": "command_not_found",
                "message": "missing",
            },
            "command_not_found",
            None,
            False,
        ),
        (
            {
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "invalid",
            },
            "invalid_arguments",
            None,
            False,
        ),
        (
            {
                "ok": False,
                "error_code": "invalid_path",
                "message": "cwd outside workspace",
            },
            "invalid_path",
            None,
            False,
        ),
    ],
)
def test_every_failed_run_command_shape_forms_attempt(
    result: dict[str, Any],
    error_code: str,
    exit_code: int | None,
    timed_out: bool,
) -> None:
    verification = VerificationState(required=True)

    attempt = verification.record_attempt(
        workspace_revision=4,
        model_step=7,
        result=result,
    )

    assert attempt.workspace_revision == 4
    assert attempt.model_step == 7
    assert attempt.succeeded is False
    assert attempt.error_code == error_code
    assert attempt.exit_code == exit_code
    assert attempt.timed_out is timed_out
    assert verification.status(4) == "unverified"


def test_read_only_success_attempt_remains_not_required() -> None:
    state = TaskState()

    state.record_run_command(model_step=1, result=SUCCESS)

    assert state.workspace_revision == 0
    assert state.verification_status == "not_required"
    assert state.verified_revision is None
    assert state.verification.last_attempt is not None
    assert state.snapshot_for_model() is None


def test_verification_snapshot_is_minimal_and_replaced() -> None:
    state = TaskState()
    context = ConversationContext(SYSTEM_PROMPT, "Change and verify a file.")
    state.record_workspace_change()
    state.record_run_command(model_step=2, result=FAILURE)

    first_snapshot = state.snapshot_for_model()
    assert first_snapshot is not None
    context.set_runtime_state(first_snapshot)
    first_system = context.messages_for_model()[0]["content"]
    runtime_state = first_system.split("Runtime task state:\n", 1)[1]
    runtime_payload = json.loads(runtime_state)
    verification_payload = runtime_payload["verification"]
    attempt_payload = verification_payload["last_attempt"]

    assert set(verification_payload) == {
        "required",
        "status",
        "workspace_revision",
        "verified_revision",
        "last_attempt",
    }
    assert set(attempt_payload) == {
        "workspace_revision",
        "model_step",
        "succeeded",
        "exit_code",
        "timed_out",
        "error_code",
    }
    assert attempt_payload["error_code"] == "command_failed"
    for forbidden in ("cwd", "stdout", "stderr", "secret output"):
        assert forbidden not in runtime_state
    assert '"workspace_revision":1' in runtime_state
    assert '"succeeded":false' in runtime_state

    state.record_workspace_change()
    context.set_runtime_state(state.snapshot_for_model())
    second_system = context.messages_for_model()[0]["content"]

    assert second_system.count("Runtime task state:") == 1
    assert '"workspace_revision":2' in second_system
    assert '"verified_revision":null' in second_system


def test_verification_snapshot_survives_complete_cycle_trimming() -> None:
    context = ConversationContext(
        "system",
        "task",
        max_chars=1_000,
        max_messages=20,
    )
    state = TaskState()
    state.record_workspace_change()
    state.record_run_command(model_step=1, result=FAILURE)
    context.set_runtime_state(state.snapshot_for_model())

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
                            "name": "run_command",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            [
                {
                    "role": "tool",
                    "tool_call_id": identifier,
                    "name": "run_command",
                    "content": "x" * 450,
                }
            ],
        )

    messages = context.messages_for_model()

    assert '"verification"' in messages[0]["content"]
    assert not any(message.get("tool_call_id") == "old" for message in messages)
    assert any(message.get("tool_call_id") == "latest" for message in messages)
    latest_assistant = next(
        message
        for message in messages
        if message.get("role") == "assistant"
    )
    assert latest_assistant["tool_calls"][0]["id"] == "latest"


def test_runner_records_invalid_run_command_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call("invalid", "run_command", '{"command":"bad"}')
                ]
            ),
            ModelResponse(text="The command was invalid."),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Try a command.")

    assert result.status == "completed"
    assert result.verification_status == "not_required"
    assert result.last_verification is not None
    assert result.last_verification.succeeded is False
    assert result.last_verification.error_code == "invalid_arguments"
    assert result.last_verification.workspace_revision == 0


def test_same_turn_attempt_order_and_mutation_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    registry = _registry_with_run_results(tmp_path, [SUCCESS, FAILURE, SUCCESS])
    calls = [
        _tool_call("write", "write_file", {"path": "value.txt", "content": "a"}),
        _tool_call("run-1", "run_command", {"command": ["check-1"]}),
        _tool_call("run-2", "run_command", {"command": ["check-2"]}),
        _tool_call("run-3", "run_command", {"command": ["check-3"]}),
    ]
    client = FakeClient(
        [ModelResponse(tool_calls=calls), ModelResponse(text="verified")]
    )

    result = AgentRunner(client, tools=registry).run("Change and verify.")

    assert result.status == "completed"
    assert result.workspace_revision == 1
    assert result.verification_status == "verified"
    assert result.verified_revision == 1
    assert result.last_verification is not None
    assert result.last_verification.succeeded is True
    assert result.last_verification.workspace_revision == 1


def test_command_before_mutation_cannot_verify_new_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    registry = _registry_with_run_results(tmp_path, [SUCCESS])
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call("run", "run_command", {"command": ["check"]}),
                    _tool_call(
                        "write",
                        "write_file",
                        {"path": "value.txt", "content": "new"},
                    ),
                ]
            ),
            ModelResponse(text="first final"),
            ModelResponse(text="honest unverified final"),
        ]
    )

    result = AgentRunner(client, tools=registry).run("Run then change.")

    assert result.status == "completed"
    assert result.workspace_revision == 1
    assert result.verification_status == "unverified"
    assert result.verified_revision is None
    assert result.last_verification is not None
    assert result.last_verification.workspace_revision == 0


def test_project_construction_attempts_follow_managed_revision_order(
    tmp_path: Path,
) -> None:
    registry = _registry_with_run_results(tmp_path, [FAILURE, SUCCESS])
    calls = [
        _tool_call("mkdir", "make_directory", {"path": "src"}),
        _tool_call(
            "write",
            "write_file",
            {"path": "src/app.txt", "content": "first"},
        ),
        _tool_call("run-failed", "run_command", {"command": ["check"]}),
        _tool_call(
            "edit",
            "edit_file",
            {
                "path": "src/app.txt",
                "old_text": "first",
                "new_text": "fixed",
            },
        ),
        _tool_call("run-success", "run_command", {"command": ["check"]}),
    ]
    client = FakeClient(
        [ModelResponse(tool_calls=calls), ModelResponse(text="verified")]
    )

    result = AgentRunner(client, tools=registry).run("Build and verify.")

    assert result.status == "completed"
    assert result.workspace_revision == 3
    assert result.verification_status == "verified"
    assert result.verified_revision == 3
    assert result.last_verification is not None
    assert result.last_verification.workspace_revision == 3
    observations = client.calls[1]["messages"][-5:]
    assert [message["tool_call_id"] for message in observations] == [
        "mkdir",
        "write",
        "run-failed",
        "edit",
        "run-success",
    ]


def test_run_command_filesystem_side_effect_is_not_managed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    registry = ToolRegistry(Workspace(tmp_path))
    definition = registry._tools["run_command"]

    def command_with_unobserved_effect(arguments: dict[str, Any]) -> dict[str, Any]:
        (tmp_path / "from-command.txt").write_text("created", encoding="utf-8")
        return SUCCESS

    registry._tools["run_command"] = ToolDefinition(
        definition.schema,
        command_with_unobserved_effect,
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call("run", "run_command", {"command": ["generator"]})
                ]
            ),
            ModelResponse(text="command finished"),
        ]
    )

    result = AgentRunner(client, tools=registry).run("Generate a file by command.")

    assert (tmp_path / "from-command.txt").exists()
    assert result.workspace_revision == 0
    assert result.verification_status == "not_required"
    assert result.last_verification is not None
    assert result.last_verification.succeeded is True


def test_completion_review_is_injected_once_then_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call(
                        "write",
                        "write_file",
                        {"path": "value.txt", "content": "new"},
                    )
                ]
            ),
            ModelResponse(text="first candidate"),
            ModelResponse(text="accepted unverified final"),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Create a file.")

    assert result.status == "completed"
    assert result.answer == "accepted unverified final"
    assert result.steps == 3
    assert result.verification_status == "unverified"
    system_messages = [call["messages"][0]["content"] for call in client.calls]
    assert "completion_review" not in system_messages[1]
    assert "completion_review" in system_messages[2]
    assert sum("completion_review" in content for content in system_messages) == 1
    assert state.pending_completion_review is None
    assert state.last_completion_review_fingerprint is not None


def test_real_state_change_allows_a_new_completion_review(
    tmp_path: Path,
) -> None:
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call(
                        "write-a",
                        "write_file",
                        {"path": "a.txt", "content": "a"},
                    )
                ]
            ),
            ModelResponse(text="candidate at revision one"),
            ModelResponse(
                tool_calls=[
                    _tool_call(
                        "write-b",
                        "write_file",
                        {"path": "b.txt", "content": "b"},
                    )
                ]
            ),
            ModelResponse(text="candidate at revision two"),
            ModelResponse(text="accepted at revision two"),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Create two files.")

    assert result.status == "completed"
    assert result.workspace_revision == 2
    system_messages = [call["messages"][0]["content"] for call in client.calls]
    assert sum("completion_review" in content for content in system_messages) == 2
    assert "completion_review" in system_messages[2]
    assert "completion_review" in system_messages[4]


def test_active_plan_and_unverified_workspace_share_one_review(
    tmp_path: Path,
) -> None:
    plan_arguments = {
        "mode": "create",
        "steps": [
            {
                "id": "build",
                "description": "Build the project",
                "status": "in_progress",
                "blocked_reason": None,
            }
        ],
    }
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call("plan", UPDATE_PLAN_ACTION_NAME, plan_arguments),
                    _tool_call(
                        "write",
                        "write_file",
                        {"path": "value.txt", "content": "new"},
                    ),
                ]
            ),
            ModelResponse(text="first candidate"),
            ModelResponse(text="honest incomplete final"),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Build a project.")

    review_system = client.calls[2]["messages"][0]["content"]
    assert '"reasons":["active_plan","unverified_workspace"]' in review_system
    assert result.status == "completed"
    assert result.plan_status == "active"
    assert [step.id for step in result.unfinished_steps] == ["build"]
    assert result.verification_status == "unverified"


def test_last_step_candidate_final_is_not_discarded(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call(
                        "write",
                        "write_file",
                        {"path": "value.txt", "content": "new"},
                    )
                ]
            ),
            ModelResponse(text="last available final"),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        max_steps=2,
    ).run("Create a file.")

    assert result.status == "completed"
    assert result.answer == "last available final"
    assert result.steps == 2
    assert result.verification_status == "unverified"
    assert len(client.calls) == 2


def test_completion_review_request_is_stable_across_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = ModelAPIError(
        "temporary_api_error",
        "temporary",
        classification="retryable",
    )
    monkeypatch.setattr(runner_module, "sleep", lambda seconds: None)
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call(
                        "write",
                        "write_file",
                        {"path": "value.txt", "content": "new"},
                    )
                ]
            ),
            ModelResponse(text="first candidate"),
            temporary,
            ModelResponse(text="accepted after retry"),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Create a file.")

    assert result.status == "completed"
    assert client.calls[2] == client.calls[3]
    assert "completion_review" in client.calls[2]["messages"][0]["content"]


@pytest.mark.parametrize(
    "terminal_response,expected_status",
    [
        (
            ModelAPIError(
                "model_api_error",
                "fatal",
                classification="fatal",
            ),
            "fatal_api_error",
        ),
        (KeyboardInterrupt(), "user_interrupt"),
        (ValueError("private runtime detail"), "runtime_error"),
    ],
)
def test_error_termination_paths_preserve_verification_facts(
    tmp_path: Path,
    terminal_response: BaseException,
    expected_status: str,
) -> None:
    root = tmp_path / expected_status
    root.mkdir()
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call(
                        "write",
                        "write_file",
                        {"path": "value.txt", "content": "new"},
                    )
                ]
            ),
            terminal_response,
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(root)),
    ).run("Create a file, then stop unexpectedly.")

    assert result.status == expected_status
    assert result.workspace_revision == 1
    assert result.verification_status == "unverified"
    assert result.verified_revision is None


def test_max_steps_preserves_verification_facts(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    _tool_call(
                        "write",
                        "write_file",
                        {"path": "value.txt", "content": "new"},
                    )
                ]
            )
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        max_steps=1,
    ).run("Create a file.")

    assert result.status == "max_steps"
    assert result.workspace_revision == 1
    assert result.verification_status == "unverified"
    assert result.verified_revision is None

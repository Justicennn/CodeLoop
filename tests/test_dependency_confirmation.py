"""Deterministic dependency-mutation classification and approval tests."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent.runner as runner_module
from codeloop.agent.runner import AgentRunner
from codeloop.agent.task_state import TaskState
from codeloop.execution.command_policy import (
    CommandApprovalRequest,
    dependency_mutation_request,
)
from codeloop.execution.tools import RUN_COMMAND_SCHEMA, ToolDefinition, ToolRegistry
from codeloop.execution.workspace import Workspace
from codeloop.interaction.approval import ConsoleCommandApprover
from codeloop.model.client import ModelResponse, ToolCall


class FakeClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append(
            {"messages": deepcopy(messages), "tools": deepcopy(tools)}
        )
        return next(self._responses)


MUTATING_COMMANDS = [
    ["pip", "install", "numpy"],
    ["pip3.11.exe", "uninstall", "numpy"],
    ["python", "-m", "pip", "install", "numpy"],
    ["python3", "-m", "pip", "uninstall", "numpy"],
    [r"C:\Python311\python.exe", "-m", "pip", "install", "numpy"],
    ["py.exe", "-3", "-m", "pip", "install", "numpy"],
    ["conda", "install", "numpy"],
    ["conda", "remove", "numpy"],
    ["conda", "update", "numpy"],
    ["uv", "add", "requests"],
    ["uv", "remove", "requests"],
    ["uv", "sync"],
    ["uv", "pip", "install", "requests"],
    ["uv", "pip", "uninstall", "requests"],
    ["uv", "pip", "sync", "requirements.txt"],
    ["poetry", "add", "requests"],
    ["poetry", "remove", "requests"],
    ["poetry", "install"],
    ["npm.cmd", "install"],
    ["NPM.CMD", "CI"],
    ["npm", "uninstall", "react"],
    ["npm", "remove", "react"],
    ["npm", "update"],
    ["npm", "ci"],
    ["pnpm", "add", "react"],
    ["pnpm", "remove", "react"],
    ["pnpm", "install"],
    ["pnpm", "update"],
    ["yarn", "add", "react"],
    ["yarn", "remove", "react"],
    ["yarn", "install"],
    ["yarn", "update"],
    ["yarn", "upgrade"],
    ["YARN.BAT", "UPGRADE"],
]


def test_approval_request_is_immutable_and_has_only_fixed_policy_fields() -> None:
    request = CommandApprovalRequest(command=("npm", "ci"))

    assert tuple(item.name for item in fields(request)) == (
        "command",
        "category",
        "reason",
    )
    assert request.category == "dependency_mutation"
    with pytest.raises(FrozenInstanceError):
        request.command = ("npm", "install")  # type: ignore[misc]
    with pytest.raises(TypeError):
        CommandApprovalRequest(  # type: ignore[call-arg]
            command=("npm", "ci"),
            category="other",
        )


@pytest.mark.parametrize("command", MUTATING_COMMANDS)
def test_explicit_dependency_mutation_forms_require_approval(
    command: list[str],
) -> None:
    request = dependency_mutation_request(command)

    assert request is not None
    assert request.command == tuple(command)
    assert request.category == "dependency_mutation"


@pytest.mark.parametrize(
    "command",
    [
        ["python", "-m", "pytest"],
        ["node", "test.js"],
        ["git", "status"],
        ["pip", "list"],
        ["pip", "show", "numpy"],
        ["pip", "freeze"],
        ["conda", "list"],
        ["npm", "list"],
        ["npm", "test"],
        ["npm", "run", "test"],
        ["uv", "pip", "list"],
        ["poetry", "show"],
        ["python", "some_script.py"],
        ["bun", "install"],
        ["cmd", "/c", "echo bun install package"],
        ["sh", "-c", "pip list"],
    ],
)
def test_read_only_ordinary_and_unlisted_forms_do_not_claim_protection(
    command: list[str],
) -> None:
    assert dependency_mutation_request(command) is None


@pytest.mark.parametrize(
    "command",
    [
        ["cmd", "/c", "pip install numpy"],
        ["cmd.exe", "/c", "echo pip install numpy"],
        ["powershell", "-Command", "pytest; npm ci"],
        ["pwsh", "-Command", "python -m pip uninstall numpy"],
        ["sh", "-c", "pytest || uv pip sync requirements.txt"],
        ["bash", "-c", "echo 'yarn add react'"],
    ],
)
def test_wrapper_scanning_is_deliberately_conservative_and_deterministic(
    command: list[str],
) -> None:
    """Wrapper scanning is deliberately conservative and deterministic;
    it does not claim shell-semantic precision.
    """
    assert dependency_mutation_request(command) is not None
    assert dependency_mutation_request(command) == dependency_mutation_request(command)


def _registry_with_safe_run_handler(
    tmp_path: Path,
    dispatched: list[dict[str, Any]],
) -> ToolRegistry:
    registry = ToolRegistry(Workspace(tmp_path))
    definition = registry._tools["run_command"]

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        dispatched.append(arguments)
        return {
            "ok": True,
            "data": {
                "command": arguments["command"],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            },
        }

    registry._tools["run_command"] = ToolDefinition(definition.schema, handler)
    return registry


def _run_command_client(command: list[str]) -> FakeClient:
    return FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "dependency-call",
                        "run_command",
                        json.dumps({"command": command}),
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )


def _latest_tool_result(client: FakeClient) -> dict[str, Any]:
    tool_message = next(
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "tool"
    )
    assert tool_message["tool_call_id"] == "dependency-call"
    return json.loads(tool_message["content"])


def test_approved_dependency_command_dispatches_only_after_callback(
    tmp_path: Path,
) -> None:
    order: list[str] = []
    dispatched: list[dict[str, Any]] = []
    registry = _registry_with_safe_run_handler(tmp_path, dispatched)
    original_handler = registry._tools["run_command"].handler

    def ordered_handler(arguments: dict[str, Any]) -> dict[str, Any]:
        order.append("dispatch")
        return original_handler(arguments)

    definition = registry._tools["run_command"]
    registry._tools["run_command"] = ToolDefinition(
        definition.schema,
        ordered_handler,
    )
    client = _run_command_client(["pip", "install", "numpy"])

    result = AgentRunner(
        client,
        tools=registry,
        on_command_approval=lambda _request: order.append("approval") or True,
    ).run("Install only if approved")

    assert result.status == "completed"
    assert order == ["approval", "dispatch"]
    assert len(dispatched) == 1
    assert _latest_tool_result(client)["ok"] is True


@pytest.mark.parametrize(
    "callback,error_code",
    [
        (lambda _request: False, "user_denied"),
        (lambda _request: 1, "user_denied"),
        (None, "approval_unavailable"),
    ],
)
def test_denied_or_unavailable_dependency_command_never_dispatches(
    tmp_path: Path,
    callback: Any,
    error_code: str,
) -> None:
    dispatched: list[dict[str, Any]] = []
    client = _run_command_client(["python", "-m", "pip", "install", "numpy"])

    result = AgentRunner(
        client,
        tools=_registry_with_safe_run_handler(tmp_path, dispatched),
        on_command_approval=callback,
    ).run("Run tests")

    observation = _latest_tool_result(client)
    assert dispatched == []
    assert observation["ok"] is False
    assert observation["error_code"] == error_code
    assert observation["data"]["category"] == "dependency_mutation"
    assert result.last_verification is None
    assert result.workspace_revision == 0


def test_approval_callback_exception_fails_closed(tmp_path: Path) -> None:
    dispatched: list[dict[str, Any]] = []
    client = _run_command_client(["npm", "ci"])

    def broken(_request: CommandApprovalRequest) -> bool:
        raise RuntimeError("presentation failed")

    result = AgentRunner(
        client,
        tools=_registry_with_safe_run_handler(tmp_path, dispatched),
        on_command_approval=broken,
    ).run("Run tests")

    assert dispatched == []
    assert _latest_tool_result(client)["error_code"] == "approval_unavailable"
    assert result.status == "completed"


def test_confirmation_eof_is_approval_unavailable(tmp_path: Path) -> None:
    dispatched: list[dict[str, Any]] = []
    client = _run_command_client(["npm", "ci"])

    def eof(_prompt: str) -> str:
        raise EOFError

    approver = ConsoleCommandApprover(read_line=eof, write_line=lambda _line: None)
    result = AgentRunner(
        client,
        tools=_registry_with_safe_run_handler(tmp_path, dispatched),
        on_command_approval=approver,
    ).run("Run tests")

    assert dispatched == []
    assert _latest_tool_result(client)["error_code"] == "approval_unavailable"
    assert result.status == "completed"


def test_confirmation_keyboard_interrupt_keeps_user_interrupt(
    tmp_path: Path,
) -> None:
    dispatched: list[dict[str, Any]] = []
    client = _run_command_client(["npm", "ci"])

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    approver = ConsoleCommandApprover(
        read_line=interrupt,
        write_line=lambda _line: None,
    )
    result = AgentRunner(
        client,
        tools=_registry_with_safe_run_handler(tmp_path, dispatched),
        on_command_approval=approver,
    ).run("Run tests")

    assert dispatched == []
    assert result.status == "user_interrupt"
    assert result.last_verification is None


def test_invalid_run_command_arguments_do_not_request_approval(
    tmp_path: Path,
) -> None:
    approvals: list[CommandApprovalRequest] = []
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "invalid-command",
                        "run_command",
                        json.dumps(
                            {
                                "command": ["pip", "install", "numpy"],
                                "unexpected": True,
                            }
                        ),
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        on_command_approval=lambda request: approvals.append(request) or True,
    ).run("Run tests")

    assert approvals == []
    tool_message = next(
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "tool"
    )
    assert json.loads(tool_message["content"])["error_code"] == "invalid_arguments"


def test_denial_does_not_replace_existing_verification_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    state.record_workspace_change()
    state.record_run_command(
        model_step=1,
        result={"ok": True, "data": {"exit_code": 0, "timed_out": False}},
    )
    previous_attempt = state.verification.last_attempt
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    client = _run_command_client(["pip", "install", "numpy"])

    result = AgentRunner(
        client,
        tools=_registry_with_safe_run_handler(tmp_path, []),
        on_command_approval=lambda _request: False,
    ).run("Run tests")

    assert state.verification.last_attempt is previous_attempt
    assert result.verification_status == "verified"
    assert result.verified_revision == 1


def test_ordinary_command_does_not_request_approval(tmp_path: Path) -> None:
    approvals: list[CommandApprovalRequest] = []
    dispatched: list[dict[str, Any]] = []
    client = _run_command_client(["python", "-m", "pytest"])

    AgentRunner(
        client,
        tools=_registry_with_safe_run_handler(tmp_path, dispatched),
        on_command_approval=lambda request: approvals.append(request) or True,
    ).run("Run tests")

    assert approvals == []
    assert len(dispatched) == 1


def test_repeated_denials_use_progress_recovery_not_repeated_failure(
    tmp_path: Path,
) -> None:
    repeated = ModelResponse(
        tool_calls=[
            ToolCall(
                "dependency-call",
                "run_command",
                json.dumps({"command": ["pip", "install", "numpy"]}),
            )
        ]
    )
    client = FakeClient(
        [repeated, repeated, repeated, ModelResponse(text="Blocked by approval.")]
    )

    result = AgentRunner(
        client,
        tools=_registry_with_safe_run_handler(tmp_path, []),
        on_command_approval=lambda _request: False,
    ).run("Run tests")

    assert result.status == "completed"
    assert result.answer == "Blocked by approval."
    assert '"progress"' in client.calls[3]["messages"][0]["content"]


def test_run_command_schema_has_no_model_controlled_approval_field() -> None:
    properties = RUN_COMMAND_SCHEMA["function"]["parameters"]["properties"]
    assert "approved" not in properties
    assert "approval" not in properties


@pytest.mark.parametrize(
    "answer,approved",
    [("y", True), ("YES", True), ("", False), ("n", False), ("later", False)],
)
def test_console_approver_is_per_command_and_defaults_to_no(
    answer: str,
    approved: bool,
) -> None:
    output: list[str] = []
    prompts: list[str] = []

    def read_line(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    request = CommandApprovalRequest(command=("npm", "ci"))
    result = ConsoleCommandApprover(
        read_line=read_line,
        write_line=output.append,
    )(request)

    assert result is approved
    assert prompts == ["Proceed? [y/N]: "]
    assert output[0] == "⚠ Dependency change"
    assert "npm ci" in output[1]
    assert request.reason in output[2]

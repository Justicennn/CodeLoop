from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent as agent_module
from codeloop.agent import AgentRunner, _FailureTracker
from codeloop.llm import ModelAPIError, ModelResponse, ToolCall
from codeloop.tools import MAX_TEXT_CHARS, ToolRegistry
from codeloop.workspace import Workspace, WorkspaceError


class FakeClient:
    def __init__(
        self,
        actions: list[ModelResponse | BaseException],
    ) -> None:
        self._actions = iter(actions)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )
        action = next(self._actions)
        if isinstance(action, BaseException):
            raise action
        return action


def test_workspace_rejects_escape(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(WorkspaceError):
        workspace.resolve("../outside.txt")
    with pytest.raises(WorkspaceError):
        workspace.resolve(str(outside))

    link = tmp_path / "escape-link.txt"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        return
    with pytest.raises(WorkspaceError):
        workspace.resolve("escape-link.txt")


def test_file_mutations_are_exact_and_never_overwrite(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    target = tmp_path / "target.py"
    target.write_bytes(
        b"def add(a, b):\r\n"
        b"    time.sleep(120)\r\n"
        b"    return a + b\r\n"
    )

    observed = registry.dispatch(
        "read_file",
        json.dumps({"path": "target.py"}),
    )
    assert observed["ok"] is True
    assert "\r" not in observed["data"]["content"]
    assert "2:     time.sleep(120)" in observed["data"]["content"]

    success = registry.dispatch(
        "edit_file",
        json.dumps(
            {
                "path": "target.py",
                "old_text": "    time.sleep(120)\n",
                "new_text": "",
            }
        ),
    )
    assert success["ok"] is True
    assert success["data"]["replacements"] == 1
    expected = b"def add(a, b):\r\n    return a + b\r\n"
    updated_bytes = target.read_bytes()
    assert updated_bytes == expected
    assert b"\n" not in updated_bytes.replace(b"\r\n", b"")

    mismatch = registry.dispatch(
        "edit_file",
        json.dumps(
            {
                "path": "target.py",
                "old_text": "missing",
                "new_text": "changed",
            }
        ),
    )
    assert mismatch["error_code"] == "edit_mismatch"
    assert target.read_bytes() == expected

    repeated = tmp_path / "repeated.txt"
    repeated.write_text("same same", encoding="utf-8")
    ambiguous = registry.dispatch(
        "edit_file",
        json.dumps(
            {
                "path": "repeated.txt",
                "old_text": "same",
                "new_text": "different",
            }
        ),
    )
    assert ambiguous["error_code"] == "edit_ambiguous"
    assert repeated.read_text(encoding="utf-8") == "same same"

    created = registry.dispatch(
        "write_file",
        json.dumps({"path": "new.txt", "content": "first"}),
    )
    rejected = registry.dispatch(
        "write_file",
        json.dumps({"path": "new.txt", "content": "second"}),
    )
    assert created["ok"] is True
    assert rejected["error_code"] == "file_exists"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "first"


def test_tool_results_are_bounded_and_secrets_are_redacted(
    tmp_path: Path,
) -> None:
    secret = "stage-three-secret"
    registry = ToolRegistry(
        Workspace(tmp_path),
        sensitive_values=(secret,),
    )
    (tmp_path / "long.txt").write_text(
        "z" * (MAX_TEXT_CHARS + 100),
        encoding="utf-8",
    )
    (tmp_path / "search.txt").write_text(
        "needle" + "y" * (MAX_TEXT_CHARS + 100),
        encoding="utf-8",
    )

    read_result = registry.dispatch(
        "read_file",
        json.dumps({"path": "long.txt"}),
    )
    search_result = registry.dispatch(
        "search_code",
        json.dumps({"query": "needle", "path": "search.txt"}),
    )
    success = registry.dispatch(
        "run_command",
        json.dumps(
            {
                "command": [
                    sys.executable,
                    "-c",
                    (
                        f"print('{secret}'); "
                        f"print('x' * {MAX_TEXT_CHARS + 100})"
                    ),
                ]
            }
        ),
    )
    failure = registry.dispatch(
        "run_command",
        json.dumps(
            {
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; print('failed', file=sys.stderr); sys.exit(3)",
                ]
            }
        ),
    )

    assert read_result["data"]["truncated"] is True
    assert search_result["data"]["truncated"] is True
    assert success["ok"] is True
    assert success["data"]["stdout_truncated"] is True
    assert success["data"]["direct_child_reaped"] is True
    assert "direct_child_terminated" not in success["data"]
    assert secret not in json.dumps(success)
    assert "[REDACTED]" in json.dumps(success)
    assert failure["error_code"] == "command_failed"
    assert failure["data"]["exit_code"] == 3
    assert "failed" in failure["data"]["stderr"]


def test_real_tool_agent_loop_returns_observation_to_model(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("hello\n", encoding="utf-8")
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-read-1",
                        name="read_file",
                        arguments='{"path":"sample.txt"}',
                    )
                ]
            ),
            ModelResponse(text="The file says hello."),
        ]
    )
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(client, tools=registry).run("Read sample.txt.")

    assert result.status == "completed"
    assert result.answer == "The file says hello."
    assert len(client.calls) == 2
    second_messages = client.calls[1]["messages"]
    assistant_message = second_messages[-2]
    tool_message = second_messages[-1]
    observation = json.loads(tool_message["content"])
    assert assistant_message["tool_calls"][0]["id"] == "call-read-1"
    assert tool_message["tool_call_id"] == "call-read-1"
    assert observation["ok"] is True
    assert "1: hello" in observation["data"]["content"]


def test_max_steps_remains_an_independent_limit(tmp_path: Path) -> None:
    repeated_call = ModelResponse(
        tool_calls=[
            ToolCall(
                id="call-repeat",
                name="list_files",
                arguments="{}",
            )
        ]
    )
    client = FakeClient([repeated_call, repeated_call, repeated_call])
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(
        client,
        tools=registry,
        max_steps=3,
    ).run("Keep listing files.")

    assert result.status == "max_steps"
    assert result.steps == 3
    assert len(client.calls) == 3


def test_command_timeout_terminates_and_reaps_direct_child(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    code = (
        "import pathlib,time; "
        "print('started', flush=True); "
        "time.sleep(2); "
        "pathlib.Path('survived.txt').write_text('bad')"
    )

    result = registry.dispatch(
        "run_command",
        json.dumps(
            {
                "command": [sys.executable, "-c", code],
                "timeout_seconds": 1,
            }
        ),
    )
    time.sleep(1.2)

    assert result["error_code"] == "command_timeout"
    assert result["data"]["timed_out"] is True
    assert result["data"]["direct_child_reaped"] is True
    assert "started" in result["data"]["stdout"]
    assert not (tmp_path / "survived.txt").exists()


def test_repeated_failure_stops_before_remaining_tool_calls(
    tmp_path: Path,
) -> None:
    final_response = ModelResponse(
        tool_calls=[
            ToolCall(
                id="third-failure",
                name="missing_tool",
                arguments='{"a":1,"b":2}',
            ),
            ToolCall(
                id="must-not-run",
                name="write_file",
                arguments='{"path":"unexpected.txt","content":"bad"}',
            ),
        ]
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="first-failure",
                        name="missing_tool",
                        arguments='{"b":2,"a":1}',
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="second-failure",
                        name="missing_tool",
                        arguments='{"a": 1, "b": 2}',
                    )
                ]
            ),
            final_response,
        ]
    )
    events: list[str] = []
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(
        client,
        tools=registry,
        on_tool_event=lambda call, _: events.append(call.id),
    ).run("Repeat a failing call.")

    assert result.status == "repeated_failure"
    assert events == ["first-failure", "second-failure", "third-failure"]
    assert not (tmp_path / "unexpected.txt").exists()


def test_failure_tracker_resets_on_every_required_change() -> None:
    tracker = _FailureTracker()
    failed_a = {"ok": False, "error_code": "error_a", "message": "failed"}
    failed_b = {"ok": False, "error_code": "error_b", "message": "failed"}
    success = {"ok": True, "data": {}}
    tool_a = ToolCall(id="a", name="tool_a", arguments='{"value":1}')
    tool_b = ToolCall(id="b", name="tool_b", arguments='{"value":1}')
    changed_arguments = ToolCall(
        id="changed",
        name="tool_b",
        arguments='{"value":2}',
    )

    assert tracker.record(tool_a, failed_a) is False
    assert tracker.record(tool_a, failed_a) is False
    assert tracker.record(tool_a, success) is False
    assert tracker.record(tool_a, failed_a) is False
    assert tracker.record(tool_a, failed_a) is False
    assert tracker.record(tool_b, failed_a) is False
    assert tracker.record(changed_arguments, failed_a) is False
    assert tracker.record(changed_arguments, failed_b) is False
    assert tracker.record(changed_arguments, failed_b) is False
    assert tracker.record(changed_arguments, failed_b) is True


def test_api_retry_classification_and_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    retryable = ModelAPIError(
        "temporary_api_error",
        "temporary",
        classification="retryable",
    )
    fatal = ModelAPIError(
        "model_api_error",
        "fatal",
        classification="fatal",
    )
    delays: list[float] = []
    monkeypatch.setattr(agent_module, "sleep", delays.append)

    recovered_client = FakeClient(
        [retryable, retryable, ModelResponse(text="recovered")]
    )
    recovered = AgentRunner(
        recovered_client,
        tools=registry,
    ).run("Retry temporary errors.")
    assert recovered.status == "completed"
    assert len(recovered_client.calls) == 3
    assert delays == [0.5, 1.0]

    delays.clear()
    fatal_client = FakeClient([fatal])
    fatal_result = AgentRunner(
        fatal_client,
        tools=registry,
    ).run("Do not retry fatal errors.")
    assert fatal_result.status == "fatal_api_error"
    assert len(fatal_client.calls) == 1
    assert delays == []

    exhausted_client = FakeClient([retryable, retryable, retryable])
    exhausted = AgentRunner(
        exhausted_client,
        tools=registry,
    ).run("Exhaust retryable errors.")
    assert exhausted.status == "fatal_api_error"
    assert len(exhausted_client.calls) == 3
    assert delays == [0.5, 1.0]


def test_interrupt_runtime_error_and_base_exception_boundaries(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))

    interrupted = AgentRunner(
        FakeClient([KeyboardInterrupt()]),
        tools=registry,
    ).run("Interrupt.")
    failed = AgentRunner(
        FakeClient([ValueError("private detail")]),
        tools=registry,
    ).run("Fail internally.")

    assert interrupted.status == "user_interrupt"
    assert failed.status == "runtime_error"
    assert "private detail" not in (failed.message or "")

    with pytest.raises(SystemExit):
        AgentRunner(
            FakeClient([SystemExit(7)]),
            tools=registry,
        ).run("Do not catch BaseException.")

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from io import StringIO
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from rich.console import Console

import codeloop.agent.runner as agent_module
from codeloop.agent.context import ConversationContext
from codeloop.agent.events import ToolEvent
from codeloop.agent.runner import AgentRunner, _FailureTracker
from codeloop.execution.tools import MAX_TEXT_CHARS, ToolRegistry
from codeloop.execution.workspace import Workspace, WorkspaceError
from codeloop.interaction.console import (
    DIFF_PREVIEW_CHARS,
    DIFF_TRUNCATION_MARKER,
    FAILURE_EVIDENCE_CHARS,
    OUTPUT_TRUNCATION_MARKER,
    SUCCESS_EVIDENCE_CHARS,
    ConsoleRenderer,
    _bounded_text,
)
from codeloop.model.client import ModelAPIError, ModelResponse, ToolCall


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

    def broken_presentation(*_args: object) -> None:
        raise RuntimeError("presentation failed")

    result = AgentRunner(
        client,
        tools=registry,
        on_tool_event=broken_presentation,
        on_model_request_started=broken_presentation,
        on_model_request_finished=broken_presentation,
    ).run("Read sample.txt.")

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


def test_context_trims_complete_cycles_and_counts_notice_in_budgets() -> None:
    def cycle(
        call_id: str,
        content: str = "ok",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        }
        results = [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "list_files",
                "content": content,
            }
        ]
        return assistant, results

    message_limited = ConversationContext(
        "system",
        "task",
        max_chars=10_000,
        max_messages=6,
    )
    for call_id in ("one", "two", "three"):
        assistant, results = cycle(call_id)
        message_limited.add_tool_cycle(assistant, results)

    assistant, results = cycle("four")
    assistant["tool_calls"].append(
        {
            "id": "four-b",
            "type": "function",
            "function": {"name": "list_files", "arguments": "{}"},
        }
    )
    results.append(
        {
            "role": "tool",
            "tool_call_id": "four-b",
            "name": "list_files",
            "content": "ok",
        }
    )
    message_limited.add_tool_cycle(assistant, results)
    messages = message_limited.messages_for_model()

    assert len(messages) == 6
    assert messages[0] == {"role": "system", "content": "system"}
    assert messages[2] == {"role": "user", "content": "task"}
    notice = json.loads(messages[1]["content"].split(": ", 1)[1])
    assert notice == {
        "conversation_history_trimmed": True,
        "guidance": "Older tool evidence is unavailable; re-read files if needed.",
        "overflow": False,
        "removed_cycles": 3,
        "removed_messages": 6,
    }
    declared_ids = [
        call["id"]
        for message in messages
        if message["role"] == "assistant"
        for call in message["tool_calls"]
    ]
    result_ids = [
        message["tool_call_id"]
        for message in messages
        if message["role"] == "tool"
    ]
    assert declared_ids == ["four", "four-b"]
    assert result_ids == declared_ids

    large_cycles = [cycle(call_id, "x" * 400) for call_id in ("a", "b", "c")]
    two_cycle_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    for assistant, results in large_cycles[:2]:
        two_cycle_messages.extend([assistant, *results])
    char_budget = len(
        json.dumps(
            two_cycle_messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    character_limited = ConversationContext(
        "system",
        "task",
        max_chars=char_budget,
        max_messages=20,
    )
    for assistant, results in large_cycles:
        character_limited.add_tool_cycle(assistant, results)
    char_messages = character_limited.messages_for_model()
    retained_ids = [
        message["tool_call_id"]
        for message in char_messages
        if message["role"] == "tool"
    ]
    serialized_chars = len(
        json.dumps(
            char_messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert retained_ids == ["c"]
    assert serialized_chars <= char_budget


def test_oversized_latest_cycle_is_preserved_for_fake_client(
    tmp_path: Path,
) -> None:
    (tmp_path / "large.txt").write_text("x" * 2_000, encoding="utf-8")
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="large-read",
                        name="read_file",
                        arguments='{"path":"large.txt"}',
                    )
                ]
            ),
            ModelResponse(text="Observed the large file."),
        ]
    )
    registry = ToolRegistry(Workspace(tmp_path))

    result = AgentRunner(
        client,
        tools=registry,
        max_context_chars=1_000,
        max_context_messages=10,
    ).run("Read the large file.")

    assert result.status == "completed"
    messages = client.calls[1]["messages"]
    notice = json.loads(messages[1]["content"].split(": ", 1)[1])
    assert notice["conversation_history_trimmed"] is False
    assert notice["overflow"] is True
    assert len(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    ) > 1_000
    assert messages[-2]["tool_calls"][0]["id"] == "large-read"
    assert messages[-1]["tool_call_id"] == "large-read"


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
        on_tool_event=lambda event: events.append(event.tool_call.id),
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


def test_failure_observation_repair_verify_loop(
    tmp_path: Path,
) -> None:
    template = (
        Path(__file__).parents[1] / "demo" / "discount_calculator"
    )
    workspace_path = tmp_path / "discount_calculator"
    copytree(template, workspace_path)
    verify_arguments = json.dumps(
        {"command": [sys.executable, "-m", "unittest", "-v"]}
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="browse",
                        name="list_files",
                        arguments="{}",
                    ),
                    ToolCall(
                        id="inspect",
                        name="read_file",
                        arguments='{"path":"discount.py"}',
                    ),
                    ToolCall(
                        id="inspect-tests",
                        name="read_file",
                        arguments='{"path":"test_discount.py"}',
                    ),
                    ToolCall(
                        id="inspect-readme",
                        name="read_file",
                        arguments='{"path":"README.md"}',
                    ),
                ],
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="baseline",
                        name="run_command",
                        arguments=verify_arguments,
                    )
                ]
            ),
            ModelResponse(
                text="The percentage must be divided by 100.",
                tool_calls=[
                    ToolCall(
                        id="repair",
                        name="edit_file",
                        arguments=json.dumps(
                            {
                                "path": "discount.py",
                                "old_text": (
                                    "discount_amount = subtotal * discount_percent"
                                ),
                                "new_text": (
                                    "discount_amount = subtotal * "
                                    "(discount_percent / 100)"
                                ),
                            }
                        ),
                    )
                ],
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="verify",
                        name="run_command",
                        arguments=verify_arguments,
                    )
                ]
            ),
            ModelResponse(
                text=(
                    "Changed discount.py to apply percentage discounts correctly. "
                    "Ran python -m unittest -v; all tests passed. "
                    "No known limitations for the requested fix.\n\n"
                    "Inline example: `discounted_total(100, 10)`.\n\n"
                    "```python\n"
                    "print('theme neutral')\n"
                    "```"
                )
            ),
        ]
    )
    registry = ToolRegistry(Workspace(workspace_path))
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=False,
            width=4_000,
        )
    )
    renderer.show_header(
        "test-model",
        workspace_path,
        "Repair the discount calculator and verify it.",
    )

    result = AgentRunner(
        client,
        tools=registry,
        on_tool_event=renderer.show_tool_event,
        on_model_request_started=renderer.start_thinking,
        on_model_request_finished=renderer.stop_thinking,
    ).run("Repair the discount calculator and verify it.")
    renderer.show_result(result)

    assert result.status == "completed"
    assert result.steps == 5
    assert result.answer is not None
    assert "discount.py" in result.answer
    assert "python -m unittest -v" in result.answer
    assert "all tests passed" in result.answer
    assert "limitations" in result.answer
    inspection = json.loads(client.calls[1]["messages"][-3]["content"])
    baseline = json.loads(client.calls[2]["messages"][-1]["content"])
    repair = json.loads(client.calls[3]["messages"][-1]["content"])
    verification = json.loads(client.calls[4]["messages"][-1]["content"])
    assert baseline["error_code"] == "command_failed"
    assert baseline["data"]["exit_code"] != 0
    assert "FAILED" in baseline["data"]["stderr"]
    assert "Traceback" in baseline["data"]["stderr"]
    assert inspection["ok"] is True
    assert "discount_amount" in inspection["data"]["content"]
    assert repair["ok"] is True
    assert verification["ok"] is True
    assert verification["data"]["exit_code"] == 0
    assert "discount_percent / 100" in (
        workspace_path / "discount.py"
    ).read_text(encoding="utf-8")
    output = rendered.getvalue()
    assert "CodeLoop · test-model" in output
    assert str(workspace_path) in output
    assert "Workspace:" not in output
    assert output.count("● Files") == 1
    assert output.index("discount.py") < output.index("README.md")
    assert output.index("README.md") < output.index("test_discount.py")
    assert "Reading discount.py" not in output
    assert "Running command" not in output
    assert "FAILED" in output
    assert "AssertionError" in output
    assert "Traceback" not in output
    assert 'File "' not in output
    assert "OK" in output
    assert "◆ discount.py" in output
    assert "-discount_amount = subtotal * discount_percent" in output
    assert "+discount_amount = subtotal * (discount_percent / 100)" in output
    assert "--- a/discount.py" not in output
    assert "+++ b/discount.py" not in output
    assert "@@" not in output
    assert "Applied 1 replacement" not in output
    assert "Done · 5 steps" in output
    assert "Changed discount.py" in output
    assert "Last command" not in output
    assert "Last successful command" not in output
    assert "unittest -v" in output
    assert "Final answer" not in output
    assert "discounted_total(100, 10)" in output
    assert "print('theme neutral')" in output
    separator_lines = [
        line
        for line in output.splitlines()
        if line and set(line) == {"─"}
    ]
    assert separator_lines == ["─" * 80]
    assert "[observation]" not in output
    assert "dispatch_duration_ms=" not in output
    assert "ms)" not in output
    assert '"ok":' not in output
    assert "baseline" not in output
    assert "Diagnosing" not in output
    assert "Planning" not in output
    assert "Running tests" not in output

    rendered.seek(0)
    rendered.truncate(0)
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="many-files",
                name="list_files",
                arguments="{}",
            ),
            result={
                "ok": True,
                "data": {
                    "entries": [
                        {"path": f"file_{index}.py", "type": "file"}
                        for index in range(6)
                    ]
                    + [{"path": "ignored", "type": "directory"}],
                },
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="duplicate-read",
                name="read_file",
                arguments='{"path":"file_0.py"}',
            ),
            result={"ok": True, "data": {"path": "file_0.py"}},
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="long-success",
                name="run_command",
                arguments="{}",
            ),
            result={
                "ok": True,
                "data": {
                    "command": ["python", "check.py"],
                    "exit_code": 0,
                    "stdout": "progress " + "x" * 600,
                    "stderr": "Ran 2 tests\nOK",
                },
            },
            dispatch_duration_ms=7,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="long-failure",
                name="run_command",
                arguments="{}",
            ),
            result={
                "ok": False,
                "error_code": "command_failed",
                "message": "Verification failed.",
                "data": {
                    "command": ["python", "check.py"],
                    "exit_code": 1,
                    "stderr": (
                        "test_demo (tests.Demo.test_demo) ... FAIL\n"
                        "========================================\n"
                        "FAIL: test_demo (tests.Demo.test_demo)\n"
                        "Traceback (most recent call last):\n"
                        "  File \"test_demo.py\", line 3, in test_demo\n"
                        "    self.assertEqual(1, 2)\n"
                        "AssertionError: 1 != 2\n"
                        "Ran 1 test in 0.001s\n"
                        "FAILED (failures=1)"
                    ),
                },
            },
            dispatch_duration_ms=8,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="generic-success",
                name="run_command",
                arguments="{}",
            ),
            result={
                "ok": True,
                "data": {
                    "command": ["tool", "build"],
                    "exit_code": 0,
                    "stdout": "fallback-one\nfallback-two\nfallback-three\nfallback-four",
                    "stderr": "",
                },
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="invalid-command",
                name="run_command",
                arguments='{"command":"python check.py"}',
            ),
            result={
                "ok": False,
                "error_code": "invalid_arguments",
                "message": "command must be a non-empty array of strings",
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="long-edit",
                name="edit_file",
                arguments=json.dumps(
                    {
                        "path": "large.py",
                        "old_text": "old_" + "a" * 2_000,
                        "new_text": "new_" + "b" * 2_000,
                    }
                ),
            ),
            result={
                "ok": True,
                "data": {"path": "large.py", "replacements": 1},
            },
            dispatch_duration_ms=1_200,
            truncated=False,
        )
    )
    bounded_output = rendered.getvalue()
    assert bounded_output.count("● Files") == 1
    assert bounded_output.count("file_0.py") == 1
    assert "file_4.py" in bounded_output
    assert "file_5.py" not in bounded_output
    assert "+1 more" in bounded_output
    assert "Command details unavailable" not in bounded_output
    assert "⚠ Command not executed · invalid_arguments" in bounded_output
    assert bounded_output.count(
        "command must be a non-empty array of strings"
    ) == 1
    assert all(
        not line.strip().startswith("Error:")
        for line in bounded_output.splitlines()
    )
    assert "Ran 2 tests" in bounded_output
    assert "OK" in bounded_output
    assert "progress " not in bounded_output
    assert "python check.py" in bounded_output
    assert "test_demo (tests.Demo.test_demo) ... FAIL" in bounded_output
    assert "AssertionError: 1 != 2" in bounded_output
    assert "FAILED (failures=1)" in bounded_output
    assert "Traceback" not in bounded_output
    assert 'File "test_demo.py"' not in bounded_output
    assert "FAIL: test_demo" not in bounded_output
    assert "fallback-one" not in bounded_output
    assert "fallback-two" in bounded_output
    assert "fallback-three" in bounded_output
    assert "fallback-four" in bounded_output
    assert "... diff truncated ..." in bounded_output
    assert "---" not in bounded_output
    assert "+++" not in bounded_output
    assert "@@" not in bounded_output
    assert "replacements" not in bounded_output
    assert "(1.2s)" not in bounded_output
    assert len(
        _bounded_text(
            "x" * (SUCCESS_EVIDENCE_CHARS + 1),
            SUCCESS_EVIDENCE_CHARS,
            OUTPUT_TRUNCATION_MARKER,
        )
    ) == SUCCESS_EVIDENCE_CHARS
    assert len(
        _bounded_text(
            "x" * (FAILURE_EVIDENCE_CHARS + 1),
            FAILURE_EVIDENCE_CHARS,
            OUTPUT_TRUNCATION_MARKER,
        )
    ) == FAILURE_EVIDENCE_CHARS
    assert len(
        _bounded_text(
            "x" * (DIFF_PREVIEW_CHARS + 1),
            DIFF_PREVIEW_CHARS,
            DIFF_TRUNCATION_MARKER,
        )
    ) == DIFF_PREVIEW_CHARS


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
    recovered_events: list[str] = []
    recovered = AgentRunner(
        recovered_client,
        tools=registry,
        on_model_request_started=lambda: recovered_events.append("started"),
        on_model_request_finished=lambda: recovered_events.append("finished"),
    ).run("Retry temporary errors.")
    assert recovered.status == "completed"
    assert len(recovered_client.calls) == 3
    assert delays == [0.5, 1.0]
    assert recovered_events == ["started", "finished"]

    delays.clear()
    fatal_client = FakeClient([fatal])
    fatal_events: list[str] = []
    fatal_result = AgentRunner(
        fatal_client,
        tools=registry,
        on_model_request_started=lambda: fatal_events.append("started"),
        on_model_request_finished=lambda: fatal_events.append("finished"),
    ).run("Do not retry fatal errors.")
    assert fatal_result.status == "fatal_api_error"
    assert len(fatal_client.calls) == 1
    assert delays == []
    assert fatal_events == ["started", "finished"]

    exhausted_client = FakeClient([retryable, retryable, retryable])
    exhausted_events: list[str] = []
    exhausted = AgentRunner(
        exhausted_client,
        tools=registry,
        on_model_request_started=lambda: exhausted_events.append("started"),
        on_model_request_finished=lambda: exhausted_events.append("finished"),
    ).run("Exhaust retryable errors.")
    assert exhausted.status == "fatal_api_error"
    assert len(exhausted_client.calls) == 3
    assert delays == [0.5, 1.0]
    assert exhausted_events == ["started", "finished"]


def test_interrupt_runtime_error_and_base_exception_boundaries(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    interrupted_events: list[str] = []
    failed_events: list[str] = []

    interrupted = AgentRunner(
        FakeClient([KeyboardInterrupt()]),
        tools=registry,
        on_model_request_started=lambda: interrupted_events.append("started"),
        on_model_request_finished=lambda: interrupted_events.append("finished"),
    ).run("Interrupt.")
    failed = AgentRunner(
        FakeClient([ValueError("private detail")]),
        tools=registry,
        on_model_request_started=lambda: failed_events.append("started"),
        on_model_request_finished=lambda: failed_events.append("finished"),
    ).run("Fail internally.")

    started_interrupt_events: list[str] = []

    def interrupt_while_starting() -> None:
        started_interrupt_events.append("started")
        raise KeyboardInterrupt()

    interrupted_before_request = AgentRunner(
        FakeClient([]),
        tools=registry,
        on_model_request_started=interrupt_while_starting,
        on_model_request_finished=lambda: started_interrupt_events.append("finished"),
    ).run("Interrupt while starting presentation.")

    assert interrupted.status == "user_interrupt"
    assert interrupted_events == ["started", "finished"]
    assert failed.status == "runtime_error"
    assert failed_events == ["started", "finished"]
    assert "private detail" not in (failed.message or "")
    assert interrupted_before_request.status == "user_interrupt"
    assert started_interrupt_events == ["started", "finished"]

    system_exit_events: list[str] = []
    with pytest.raises(SystemExit):
        AgentRunner(
            FakeClient([SystemExit(7)]),
            tools=registry,
            on_model_request_started=lambda: system_exit_events.append("started"),
            on_model_request_finished=lambda: system_exit_events.append("finished"),
        ).run("Do not catch BaseException.")
    assert system_exit_events == ["started", "finished"]

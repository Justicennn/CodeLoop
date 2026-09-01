from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from importlib.resources import files
from inspect import signature
from io import StringIO
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from rich.console import Console

import codeloop.agent.runner as agent_module
import codeloop.execution.tools as tools_module
import codeloop.interaction.console as console_module
from codeloop.agent.context import ConversationContext
from codeloop.agent.events import (
    CoreActionEvent,
    RecoveryEvent,
    ReviewFindingProjection,
    ToolEvent,
)
from codeloop.agent.plan import PlanStep
from codeloop.prompts import SYSTEM_PROMPT
from codeloop.agent.runner import (
    DEFAULT_MAX_STEPS,
    AgentResult,
    AgentRunner,
    _FailureTracker,
)
from codeloop.execution.tools import MAX_TEXT_CHARS, ToolRegistry
from codeloop.execution.workspace import Workspace, WorkspaceError
from codeloop.interaction.console import (
    FAILURE_EVIDENCE_CHARS,
    OUTPUT_TRUNCATION_MARKER,
    SUCCESS_EVIDENCE_CHARS,
    ConsoleRenderer,
    _CODELOOP_MARKDOWN_STYLES,
    _bounded_text,
    _get_content_width,
    _get_horizontal_margin,
    _get_safe_terminal_width,
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


def test_core_action_event_contains_only_its_bounded_projection(
    tmp_path: Path,
) -> None:
    events: list[CoreActionEvent] = []
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="requirements",
                        name="update_requirements",
                        arguments='{"requirements":[]}',
                    )
                ]
            ),
            ModelResponse(text="No source requirements were needed."),
        ]
    )

    def broken_callback(event: CoreActionEvent) -> None:
        events.append(event)
        raise RuntimeError("presentation failed")

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
        on_core_action_event=broken_callback,
    ).run("Inspect only.")

    assert result.status == "completed"
    assert len(events) == 1
    event = events[0]
    assert event.name == "update_requirements"
    assert event.call_id == "requirements"
    assert event.result["ok"] is True
    assert event.requirement_count == 0
    assert event.requirement_sources == ()
    assert event.plan_steps is None
    assert event.review_findings is None
    assert not hasattr(event, "task_state")
    assert not hasattr(event, "context")
    assert not hasattr(event, "tool_call")


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
        "removed_previous_messages": 0,
        "removed_previous_pairs": 0,
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


def test_default_max_steps_is_shared_and_raised_to_thirty() -> None:
    assert DEFAULT_MAX_STEPS == 30
    assert signature(AgentRunner).parameters["max_steps"].default == 30


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


def test_run_command_decodes_utf8_chinese_stdout_and_stderr(
    tmp_path: Path,
) -> None:
    stdout_text = "中文标准输出"
    stderr_text = "中文错误输出"
    code = (
        "import sys; "
        f"sys.stdout.buffer.write({stdout_text.encode('utf-8')!r}); "
        f"sys.stderr.buffer.write({stderr_text.encode('utf-8')!r})"
    )

    result = ToolRegistry(Workspace(tmp_path)).dispatch(
        "run_command",
        json.dumps({"command": [sys.executable, "-c", code]}),
    )

    assert result["ok"] is True
    assert result["data"]["stdout"] == stdout_text
    assert result["data"]["stderr"] == stderr_text


def test_command_output_decode_falls_back_to_local_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools_module,
        "_local_command_output_encodings",
        lambda: ("gbk",),
    )
    expected = "本地编码输出"

    assert tools_module._decode_command_output(expected.encode("gbk")) == expected


def test_command_output_decode_replaces_unknown_bytes_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools_module,
        "_local_command_output_encodings",
        lambda: (),
    )

    assert tools_module._decode_command_output(b"ascii-then-\xff") == (
        "ascii-then-\ufffd"
    )


def test_run_command_preserves_ascii_output(tmp_path: Path) -> None:
    code = (
        "import sys; "
        "sys.stdout.buffer.write(b'ascii stdout'); "
        "sys.stderr.buffer.write(b'ascii stderr')"
    )

    result = ToolRegistry(Workspace(tmp_path)).dispatch(
        "run_command",
        json.dumps({"command": [sys.executable, "-c", code]}),
    )

    assert result["ok"] is True
    assert result["data"]["stdout"] == "ascii stdout"
    assert result["data"]["stderr"] == "ascii stderr"


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
        on_core_action_event=renderer.show_core_action_event,
        on_recovery_event=renderer.show_recovery_event,
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
    assert "● Files" not in output
    assert "README.md" not in output
    assert "test_discount.py" not in output
    assert "Reading discount.py" not in output
    assert "Running command" not in output
    assert "FAILED" in output
    assert "AssertionError" in output
    assert "Traceback" not in output
    assert 'File "' not in output
    assert "OK" in output
    assert "M Updated discount.py · 1 replacement" in output
    assert "-discount_amount = subtotal * discount_percent" not in output
    assert "+discount_amount = subtotal * (discount_percent / 100)" not in output
    assert "--- a/discount.py" not in output
    assert "+++ b/discount.py" not in output
    assert "@@" not in output
    assert "Applied 1 replacement" not in output
    assert "DONE" in output
    assert "Task completed · 5 steps" in output
    assert "VERIFICATION" in output
    assert output.count("Changed discount.py") == 1
    assert "Last command" not in output
    assert "Last successful command" not in output
    assert "unittest -v" in output
    assert "Final answer" not in output
    assert "discounted_total(100, 10)" in output
    assert "print('theme neutral')" in output
    assert "WORKING" in output
    assert "[observation]" not in output
    assert "dispatch_duration_ms=" not in output
    assert "ms)" not in output
    assert '"ok":' not in output
    assert "baseline" not in output
    assert "DIAGNOSIS" not in output
    assert "REPAIR" not in output
    assert "Diagnosing" not in output
    assert "Planning" not in output
    assert "Running tests" not in output

    rendered.seek(0)
    rendered.truncate(0)
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="overview",
                name="repository_overview",
                arguments="{}",
            ),
            result={
                "ok": True,
                "data": {
                    "path": ".",
                    "anchors": {"items": ["AGENTS.md"]},
                },
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
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
                    "stderr": "Ran 2 tests\n31 passed · 0 failed\nOK",
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
                "data": {
                    "path": "large.py",
                    "replacements": 1,
                    "workspace_changed": True,
                },
            },
            dispatch_duration_ms=1_200,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="failed-read-id",
                name="read_file",
                arguments='{"path":"missing.py"}',
            ),
            result={
                "ok": False,
                "error_code": "file_not_found",
                "message": "Path does not exist: missing.py",
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="mkdir-created-id",
                name="make_directory",
                arguments='{"path":"assets"}',
            ),
            result={
                "ok": True,
                "data": {"path": "assets", "workspace_changed": True},
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="mkdir-ready-id",
                name="make_directory",
                arguments='{"path":"existing"}',
            ),
            result={
                "ok": True,
                "data": {"path": "existing", "workspace_changed": False},
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="write-id",
                name="write_file",
                arguments='{"path":"new.py","content":"x"}',
            ),
            result={
                "ok": True,
                "data": {"path": "new.py", "workspace_changed": True},
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="unchanged-edit-id",
                name="edit_file",
                arguments='{"path":"same.py","old_text":"x","new_text":"x"}',
            ),
            result={
                "ok": True,
                "data": {
                    "path": "same.py",
                    "replacements": 1,
                    "workspace_changed": False,
                },
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    bounded_output = rendered.getvalue()
    assert "● Files" not in bounded_output
    assert "file_0.py" not in bounded_output
    assert "file_4.py" not in bounded_output
    assert "file_5.py" not in bounded_output
    assert "repository_overview" not in bounded_output
    assert "AGENTS.md" not in bounded_output
    assert "✗ read_file · missing.py · file_not_found" in bounded_output
    assert "Path does not exist: missing.py" in bounded_output
    assert "✓ Created assets" in bounded_output
    assert "✓ Directory ready existing" in bounded_output
    assert "+ Created new.py" in bounded_output
    assert "M Updated new.py" not in bounded_output
    assert "✓ Unchanged same.py" in bounded_output
    assert "Command details unavailable" not in bounded_output
    assert "⚠ run_command · invalid_arguments" in bounded_output
    assert bounded_output.count(
        "command must be a non-empty array of strings"
    ) == 1
    assert all(
        not line.strip().startswith("Error:")
        for line in bounded_output.splitlines()
    )
    assert "Ran 2 tests" not in bounded_output
    assert "31 passed · 0 failed" in bounded_output
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
    assert "fallback-two" not in bounded_output
    assert "fallback-three" not in bounded_output
    assert "fallback-four" not in bounded_output
    assert "---" not in bounded_output
    assert "+++" not in bounded_output
    assert "@@" not in bounded_output
    assert "M Updated large.py · 1 replacement" in bounded_output
    assert "(1.2s)" not in bounded_output
    assert "workspace_changed" not in bounded_output
    assert "write-id" not in bounded_output
    assert '"ok"' not in bounded_output
    assert "Thought" not in bounded_output
    assert "Reason" not in bounded_output
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


def test_compact_result_uses_real_verification_and_termination_status() -> None:
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=False,
            width=200,
        )
    )

    renderer.show_result(
        AgentResult(
            status="completed",
            answer="Model final answer.",
            steps=3,
            verification_status="unverified",
        )
    )
    unverified = rendered.getvalue()
    assert "✓ Task completed · 3 steps" in unverified
    assert "VERIFICATION" in unverified
    assert "⚠ Managed changes are not verified" in unverified
    assert "Model final answer." in unverified

    rendered.seek(0)
    rendered.truncate(0)
    renderer.show_result(
        AgentResult(
            status="completed",
            answer="Read-only final.",
            steps=1,
            verification_status="not_required",
        )
    )
    not_required = rendered.getvalue()
    assert "✓ Task completed · 1 steps" in not_required
    assert "Verified" not in not_required
    assert "Unverified" not in not_required

    rendered.seek(0)
    rendered.truncate(0)
    renderer.show_result(
        AgentResult(
            status="no_progress",
            answer=None,
            steps=8,
            message="No material progress was detected.",
        )
    )
    stopped = rendered.getvalue()
    assert "STOPPED" in stopped
    assert "✗ no_progress" in stopped
    assert "No material progress was detected." in stopped
    assert "Done" not in stopped


def test_responsive_presentation_widths_share_pure_bounded_geometry() -> None:
    assert _get_safe_terminal_width(1) == 1
    assert _get_safe_terminal_width(80) == 79

    safe_widths = (79, 139, 399)
    content_widths = tuple(_get_content_width(width) for width in safe_widths)
    assert content_widths[0] == safe_widths[0]
    assert content_widths[0] < content_widths[1] < content_widths[2]
    assert content_widths[1] != 104

    for safe_width, content_width in zip(safe_widths, content_widths):
        assert 1 <= content_width <= safe_width
        left, right = _get_horizontal_margin(safe_width, content_width)
        assert left + content_width + right == safe_width
        assert left == 0
        assert right == safe_width - content_width


def test_final_markdown_headings_use_focused_brand_style() -> None:
    for level in range(1, 7):
        style = _CODELOOP_MARKDOWN_STYLES[f"markdown.h{level}"]
        assert style.color is not None
        assert style.color.name == "orange3"
        assert style.bold is True
        assert style.underline is False
        assert style.reverse is False

    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=False,
            width=80,
        )
    )
    renderer.show_result(
        AgentResult(
            status="completed",
            answer="# Left aligned heading\n\nBody.",
            steps=1,
        )
    )
    heading_line = next(
        line for line in rendered.getvalue().splitlines() if "Left aligned" in line
    )
    assert heading_line.startswith("Left aligned heading")
    body_line = next(
        line for line in rendered.getvalue().splitlines() if line.strip() == "Body."
    )
    assert body_line.startswith("Body.")


def test_owned_tty_input_redraw_is_single_attempt_and_bounded() -> None:
    class InputConsole(Console):
        def __init__(self) -> None:
            super().__init__(
                file=StringIO(),
                color_system=None,
                force_terminal=True,
                width=100,
            )
            self.controls: list[object] = []

        def input(
            self,
            prompt: object = "",
            *args: object,
            **kwargs: object,
        ) -> str:
            del args, kwargs
            self.print(prompt, end="")
            return "Review this project"

        def control(self, *control: object) -> None:
            self.controls.extend(control)

    console = InputConsole()
    renderer = ConsoleRenderer(console=console)
    renderer.show_input_top_rule()
    assert renderer.read_user_input() == "Review this project"
    renderer.show_submitted_user_message("Review this project")

    assert len(console.controls) == 1
    assert "❯ Review this project" in console.file.getvalue()

    unsafe_console = InputConsole()
    unsafe_renderer = ConsoleRenderer(console=unsafe_console)
    unsafe_renderer.show_input_top_rule()
    unsafe_renderer.read_user_input()
    unsafe_renderer._input_had_presentation_output = True
    unsafe_renderer.show_submitted_user_message("Review this project")

    assert unsafe_console.controls == []
    assert "❯ Review this project" not in unsafe_console.file.getvalue()


def test_live_presentation_is_transient_event_driven_and_final_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []

    class FakeLive:
        instances: list["FakeLive"] = []

        def __init__(
            self,
            renderable: object,
            *,
            console: Console,
            transient: bool,
            auto_refresh: bool,
        ) -> None:
            del console
            self.renderable = renderable
            self.transient = transient
            self.auto_refresh = auto_refresh
            self.updates = 0
            self.refreshes = 0
            self.stops = 0
            self.instances.append(self)

        def start(self, *, refresh: bool) -> None:
            assert refresh is False
            lifecycle.append("start")

        def update(self, renderable: object, *, refresh: bool) -> None:
            assert refresh is False
            self.renderable = renderable
            self.updates += 1
            lifecycle.append("update")

        def refresh(self) -> None:
            self.refreshes += 1
            lifecycle.append("refresh")

        def stop(self) -> None:
            self.stops += 1
            lifecycle.append("stop")

    monkeypatch.setattr(console_module, "Live", FakeLive)
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=True,
            width=100,
        ),
        live=True,
    )

    renderer.start_thinking()
    live = FakeLive.instances[0]
    assert live.transient is True
    assert live.auto_refresh is False
    assert live.refreshes == 1
    assert renderer._presentation.snapshot().phase is None

    renderer.show_narration("我先检查当前项目。")
    narration_snapshot = renderer._presentation.snapshot()
    assert narration_snapshot.phase is None
    assert narration_snapshot.current == "我先检查当前项目。"
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall("read", "read_file", '{"path":"app.py"}'),
            result={"ok": True, "data": {"path": "app.py"}},
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    snapshot = renderer._presentation.snapshot()
    assert snapshot.phase == "Inspecting workspace"
    assert snapshot.current is None
    assert len(snapshot.actions) == 1
    assert snapshot.actions[0].target == "app.py"
    assert live.updates == 2
    assert live.refreshes == 3
    renderer.start_thinking()
    assert renderer._presentation.snapshot().phase == "Inspecting workspace"
    assert live.updates == 2
    assert live.refreshes == 3

    renderer.show_result(
        AgentResult(status="completed", answer="FINAL_ONLY", steps=2)
    )
    output = rendered.getvalue()
    assert live.stops == 1
    assert lifecycle.index("stop") < len(lifecycle)
    assert output.count("FINAL_ONLY") == 1
    assert "DONE" not in output
    assert "Task completed" not in output
    assert "app.py" not in output
    renderer.close()
    assert live.stops == 1

    stopped_output = StringIO()
    stopped_renderer = ConsoleRenderer(
        console=Console(
            file=stopped_output,
            color_system=None,
            force_terminal=True,
            width=100,
        ),
        live=True,
    )
    stopped_renderer.start_thinking()
    stopped_live = FakeLive.instances[1]
    stopped_renderer.show_result(
        AgentResult(
            status="no_progress",
            answer=None,
            steps=3,
            message="No material progress was detected.",
        )
    )
    stopped_text = stopped_output.getvalue()
    assert stopped_live.stops == 1
    assert "Stopped · no_progress" in stopped_text
    assert "No material progress was detected." in stopped_text
    assert "STOPPED" not in stopped_text


def test_live_failure_falls_back_without_reconsuming_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingLive:
        def __init__(self, renderable: object, **_kwargs: object) -> None:
            del renderable
            self.stops = 0

        def start(self, *, refresh: bool) -> None:
            assert refresh is False

        def update(self, renderable: object, *, refresh: bool) -> None:
            del renderable, refresh
            raise RuntimeError("render failed")

        def refresh(self) -> None:
            pass

        def stop(self) -> None:
            self.stops += 1

    monkeypatch.setattr(console_module, "Live", FailingLive)
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=True,
            width=100,
        ),
        live=True,
    )
    renderer.start_thinking()
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall("edit", "edit_file", '{"path":"app.py"}'),
            result={
                "ok": True,
                "data": {
                    "path": "app.py",
                    "workspace_changed": True,
                    "replacements": 1,
                },
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )
    snapshot = renderer._presentation.snapshot()
    assert len(snapshot.actions) == 1
    assert snapshot.actions[0].count == 1
    assert rendered.getvalue().count("Updated app.py") == 1

    renderer.show_result(
        AgentResult(status="completed", answer="fallback final", steps=2)
    )
    output = rendered.getvalue()
    assert "DONE" in output
    assert output.count("fallback final") == 1


def test_structured_sections_render_only_explicit_runtime_facts() -> None:
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=False,
            width=80,
        ),
        live=True,
    )
    renderer.show_core_action_event(
        CoreActionEvent(
            name="update_requirements",
            call_id="requirements",
            result={"ok": True, "data": {"changed": True}},
            requirement_count=1,
            requirement_sources=("requirements.md",),
        )
    )
    renderer.show_core_action_event(
        CoreActionEvent(
            name="update_plan",
            call_id="plan",
            result={"ok": True, "data": {"changed": True}},
            plan_steps=(
                PlanStep(id="one", description="Inspect", status="completed"),
                PlanStep(id="two", description="Implement", status="in_progress"),
                PlanStep(id="three", description="Verify", status="pending"),
            ),
        )
    )
    renderer.show_core_action_event(
        CoreActionEvent(
            name="update_plan",
            call_id="plan-update",
            result={"ok": True, "data": {"changed": True}},
            plan_steps=(
                PlanStep(id="one", description="Inspect", status="completed"),
                PlanStep(id="two", description="Implement", status="completed"),
                PlanStep(id="three", description="Verify", status="in_progress"),
            ),
        )
    )
    renderer.show_core_action_event(
        CoreActionEvent(
            name="update_review_findings",
            call_id="review",
            result={"ok": True, "data": {"changed": True}},
            review_findings=(
                ReviewFindingProjection(
                    finding_type="issue",
                    title="Explicit finding",
                    priority="high",
                ),
            ),
        )
    )
    for identifier, name, path in (
        ("create-a", "write_file", "a.py"),
        ("edit-a", "edit_file", "a.py"),
        ("edit-b", "edit_file", "b.py"),
    ):
        renderer.show_tool_event(
            ToolEvent(
                tool_call=ToolCall(
                    id=identifier,
                    name=name,
                    arguments=json.dumps({"path": path}),
                ),
                result={
                    "ok": True,
                    "data": {"path": path, "workspace_changed": True},
                },
                dispatch_duration_ms=1,
                truncated=False,
            )
        )
    renderer.show_result(
        AgentResult(status="completed", answer="FINAL_ONCE", steps=4)
    )

    output = rendered.getvalue()
    assert "UNDERSTANDING" in output
    assert "Source: requirements.md" in output
    assert "1 requirement registered" in output
    assert "PLAN" in output
    assert "✓ Inspect" in output
    assert "● Implement" in output
    assert "○ Verify" in output
    assert output.count("Inspect") == 1
    assert "REVIEW" in output
    assert "HIGH · issue · Explicit finding" in output
    assert "CHANGED · 2 files" in output
    assert "+ created · a.py" in output
    assert "M modified · b.py" in output
    assert output.count("FINAL_ONCE") == 1
    assert "DONE" in output
    assert "DIAGNOSIS" not in output
    assert "REPAIR" not in output


@pytest.mark.parametrize("width", [80, 140])
def test_renderer_width_and_explicit_recovery_are_safe(width: int) -> None:
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=False,
            width=width,
        )
    )
    renderer.show_recovery_event(RecoveryEvent())
    renderer.show_result(
        AgentResult(status="no_progress", answer=None, steps=4)
    )

    output = rendered.getvalue()
    assert "REPAIR" in output
    assert "Recovery requested after no material progress" in output
    assert "STOPPED" in output


def test_denied_command_is_working_not_verification() -> None:
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=False,
            width=100,
        )
    )
    renderer.show_tool_event(
        ToolEvent(
            tool_call=ToolCall(
                id="denied",
                name="run_command",
                arguments="{}",
            ),
            result={
                "ok": False,
                "error_code": "user_denied",
                "message": "The user did not approve it.",
            },
            dispatch_duration_ms=1,
            truncated=False,
        )
    )

    output = rendered.getvalue()
    assert "WORKING" in output
    assert "VERIFICATION" not in output


def test_public_narration_and_detailed_final_are_rendered_without_inference() -> None:
    rendered = StringIO()
    renderer = ConsoleRenderer(
        console=Console(
            file=rendered,
            color_system=None,
            force_terminal=False,
            width=4_000,
        )
    )
    narration = "我先检查购买流程；如果验证失败，再调整找零策略。"
    renderer.show_narration(narration)
    detailed_final = "详细结果：" + "保留完整上下文。" * 300 + "FINAL_TAIL"
    renderer.show_result(
        AgentResult(status="completed", answer=detailed_final, steps=2)
    )

    output = rendered.getvalue()
    assert output.count(narration) == 1
    assert "Inspecting..." not in output
    assert "Diagnosing..." not in output
    assert "Planning..." not in output
    assert "Thinking..." not in output
    assert "详细结果：" in output
    assert "FINAL_TAIL" in output


def test_system_prompt_locks_optional_narration_and_answer_scope() -> None:
    prompt = SYSTEM_PROMPT
    resource_prompt = files("codeloop.prompts").joinpath("system.md").read_text(
        encoding="utf-8"
    )
    assert prompt == resource_prompt
    assert prompt
    for heading in (
        "# CodeLoop 系统提示词",
        "## Workspace 与证据",
        "## 需求来源",
        "## 视觉来源",
        "## 验证",
        "## 最终回答",
    ):
        assert heading in prompt
    assert "公开叙述是可选的" in prompt
    assert "绝不能提供 private reasoning" in prompt
    assert "完整意图和会话上下文" in prompt
    assert "绝不能通过关键词或短语匹配" in prompt
    assert "纯 Review 类任务" in prompt
    assert "两到四项" in prompt
    assert "多个显式主要子目标" in prompt
    assert "实际修改" in prompt
    assert "最强的真实验证结果" in prompt
    assert "用户明确要求细节时应充分展开" in prompt
    assert "依赖和环境变更由用户控制" in prompt
    assert "user_denied" in prompt
    assert "approval_unavailable" in prompt
    assert "简单局部任务不要求 overview 或 working set" in prompt


def test_runner_exposes_the_current_action_schema_order(
    tmp_path: Path,
) -> None:
    client = FakeClient([ModelResponse(text="done")])

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Answer a simple question")

    assert result.status == "completed"
    assert [
        schema["function"]["name"] for schema in client.calls[0]["tools"]
    ] == [
        "update_plan",
        "update_requirements",
        "update_working_set",
        "update_review_findings",
        "repository_overview",
        "list_files",
        "read_file",
        "read_document",
        "read_webpage",
        "read_image",
        "search_code",
        "edit_file",
        "write_file",
        "make_directory",
        "run_command",
    ]


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

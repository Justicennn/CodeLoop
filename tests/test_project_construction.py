"""Stage 7B project-construction and managed mutation-effect tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import codeloop.agent.runner as runner_module
from codeloop.agent.plan import UPDATE_PLAN_ACTION_NAME
from codeloop.agent.runner import AgentRunner
from codeloop.agent.task_state import TaskState
from codeloop.execution.tools import ToolDefinition, ToolRegistry
from codeloop.execution.workspace import Workspace, WorkspaceError
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
        self.calls.append({"messages": messages, "tools": tools})
        return next(self._responses)


def _dispatch(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return registry.dispatch(name, json.dumps(arguments))


def test_workspace_root_must_exist_and_make_directory_never_creates_it(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-root"

    with pytest.raises(WorkspaceError) as exc_info:
        Workspace(missing_root)

    assert exc_info.value.error_code == "invalid_workspace"
    assert not missing_root.exists()


def test_make_directory_nested_existing_and_registered_schema(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(workspace)
    original_root = workspace.root
    (tmp_path / "existing").mkdir()

    single = _dispatch(registry, "make_directory", {"path": "single"})
    created = _dispatch(registry, "make_directory", {"path": "src/app/core"})
    partial_parent = _dispatch(
        registry,
        "make_directory",
        {"path": "existing/child"},
    )
    existing = _dispatch(registry, "make_directory", {"path": "src/app/core"})
    root_noop = _dispatch(registry, "make_directory", {"path": "."})

    assert single["data"]["created_directories"] == ["single"]
    assert created == {
        "ok": True,
        "data": {
            "path": "src/app/core",
            "created_directories": ["src", "src/app", "src/app/core"],
            "created_count": 3,
            "workspace_changed": True,
        },
    }
    assert partial_parent["data"]["created_directories"] == ["existing/child"]
    assert existing["ok"] is True
    assert existing["data"]["created_directories"] == []
    assert existing["data"]["workspace_changed"] is False
    assert root_noop["ok"] is True
    assert root_noop["data"]["path"] == "."
    assert root_noop["data"]["workspace_changed"] is False
    assert workspace.root == original_root
    assert len(registry.schemas) == 11
    assert "repository_overview" in registry.names
    assert "make_directory" in registry.names


def test_make_directory_rejects_conflicts_and_workspace_escape(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    direct_conflict = _dispatch(
        registry,
        "make_directory",
        {"path": "file.txt"},
    )
    parent_conflict = _dispatch(
        registry,
        "make_directory",
        {"path": "file.txt/child"},
    )
    traversal = _dispatch(registry, "make_directory", {"path": "../outside"})
    absolute = _dispatch(
        registry,
        "make_directory",
        {"path": str(tmp_path.parent / "outside")},
    )

    assert direct_conflict["error_code"] == "path_conflict"
    assert parent_conflict["error_code"] == "path_conflict"
    assert traversal["error_code"] == "invalid_path"
    assert absolute["error_code"] == "invalid_path"
    for result in (direct_conflict, parent_conflict, traversal, absolute):
        assert result["ok"] is False
        assert result["data"]["workspace_changed"] is False


def test_make_directory_internal_symlink_and_external_escape(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    internal_target = workspace_root / "real"
    internal_target.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(
            internal_target,
            workspace_root / "inside-link",
            target_is_directory=True,
        )
        os.symlink(
            outside,
            workspace_root / "outside-link",
            target_is_directory=True,
        )
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable on this platform")

    registry = ToolRegistry(Workspace(workspace_root))
    internal = _dispatch(
        registry,
        "make_directory",
        {"path": "inside-link/child"},
    )
    escaped = _dispatch(
        registry,
        "make_directory",
        {"path": "outside-link/child"},
    )

    assert internal["ok"] is True
    assert internal["data"]["path"] == "real/child"
    assert internal["data"]["workspace_changed"] is True
    assert (internal_target / "child").is_dir()
    assert escaped["ok"] is False
    assert escaped["error_code"] == "invalid_path"
    assert escaped["data"]["workspace_changed"] is False
    assert not (outside / "child").exists()


def test_partial_make_directory_failure_reports_confirmed_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    original_mkdir = Path.mkdir

    def fail_second_level(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "app":
            raise OSError("simulated local failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_second_level)
    result = _dispatch(registry, "make_directory", {"path": "src/app"})

    assert result["ok"] is False
    assert result["error_code"] == "tool_error"
    assert result["data"] == {
        "path": "src/app",
        "created_directories": ["src"],
        "created_count": 1,
        "workspace_changed": True,
    }
    assert (tmp_path / "src").is_dir()
    assert not (tmp_path / "src" / "app").exists()
    assert registry.confirmed_workspace_change("make_directory", result) is True


def test_write_and_edit_report_real_mutation_and_noop(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))

    written = _dispatch(
        registry,
        "write_file",
        {"path": "value.txt", "content": "alpha\n"},
    )
    write_failure = _dispatch(
        registry,
        "write_file",
        {"path": "value.txt", "content": "again"},
    )
    edited = _dispatch(
        registry,
        "edit_file",
        {"path": "value.txt", "old_text": "alpha", "new_text": "beta"},
    )
    edit_noop = _dispatch(
        registry,
        "edit_file",
        {"path": "value.txt", "old_text": "beta", "new_text": "beta"},
    )
    edit_failure = _dispatch(
        registry,
        "edit_file",
        {"path": "value.txt", "old_text": "missing", "new_text": "x"},
    )

    assert written["data"]["workspace_changed"] is True
    assert write_failure["ok"] is False
    assert write_failure["data"]["workspace_changed"] is False
    assert edited["data"]["workspace_changed"] is True
    assert edit_noop["ok"] is True
    assert edit_noop["data"]["workspace_changed"] is False
    assert edit_failure["ok"] is False
    assert edit_failure["data"]["workspace_changed"] is False
    assert (tmp_path / "value.txt").read_text(encoding="utf-8") == "beta\n"


def test_write_failure_after_exclusive_create_reports_partial_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    original_open = Path.open

    class FailingWriter:
        def __init__(self, output: Any) -> None:
            self._output = output

        def __enter__(self) -> "FailingWriter":
            return self

        def write(self, content: bytes) -> int:
            self._output.write(content[:1])
            self._output.flush()
            raise OSError("simulated write failure")

        def __exit__(self, *args: object) -> None:
            self._output.close()

    def failing_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        output = original_open(path, mode, *args, **kwargs)
        if path.name == "partial.txt" and mode == "xb":
            return FailingWriter(output)
        return output

    monkeypatch.setattr(Path, "open", failing_open)
    result = _dispatch(
        registry,
        "write_file",
        {"path": "partial.txt", "content": "content"},
    )

    assert result["ok"] is False
    assert result["error_code"] == "tool_error"
    assert result["data"]["workspace_changed"] is True
    assert (tmp_path / "partial.txt").exists()
    assert registry.confirmed_workspace_change("write_file", result) is True


def test_registry_trusts_only_managed_mutation_effects(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    forged = {"ok": True, "data": {"workspace_changed": True}}
    read_definition = registry._tools["read_file"]
    registry._tools["read_file"] = ToolDefinition(
        read_definition.schema,
        lambda arguments: forged,
    )

    result = registry.dispatch("read_file", '{"path":"ignored"}')

    assert result["data"]["workspace_changed"] is True
    assert registry.confirmed_workspace_change("read_file", result) is False
    for tool_name in (
        "list_files",
        "read_file",
        "read_document",
        "read_webpage",
        "read_image",
        "search_code",
        "run_command",
    ):
        assert registry.confirmed_workspace_change(tool_name, forged) is False
    assert registry.confirmed_workspace_change("unknown", forged) is False


def test_task_state_revision_records_all_four_orthogonal_results() -> None:
    state = TaskState()
    outcomes = [
        ({"ok": True, "data": {"workspace_changed": True}}, 1),
        ({"ok": True, "data": {"workspace_changed": False}}, 1),
        ({"ok": False, "data": {"workspace_changed": False}}, 1),
        ({"ok": False, "data": {"workspace_changed": True}}, 2),
    ]

    assert state.workspace_revision == 0
    for result, expected_revision in outcomes:
        if result["data"]["workspace_changed"] is True:
            state.record_workspace_change()
        assert state.workspace_revision == expected_revision


def test_runner_does_not_increment_for_success_noop_or_unchanged_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    conflict = tmp_path / "conflict"
    conflict.write_text("file", encoding="utf-8")
    registry = ToolRegistry(Workspace(tmp_path))

    noop_state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: noop_state)
    noop_result = AgentRunner(
        FakeClient(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="noop",
                            name="make_directory",
                            arguments='{"path":"existing"}',
                        )
                    ]
                ),
                ModelResponse(text="done"),
            ]
        ),
        tools=registry,
    ).run("Keep the existing directory.")

    failed_state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: failed_state)
    failed_result = AgentRunner(
        FakeClient(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="conflict",
                            name="make_directory",
                            arguments='{"path":"conflict"}',
                        )
                    ]
                ),
                ModelResponse(text="cannot create directory"),
            ]
        ),
        tools=registry,
    ).run("Try a conflicting directory target.")

    assert noop_result.status == "completed"
    assert noop_state.workspace_revision == 0
    assert failed_result.status == "completed"
    assert failed_state.workspace_revision == 0


def test_runner_keeps_plan_and_workspace_revisions_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    calls = [
        ToolCall(
            id="plan",
            name=UPDATE_PLAN_ACTION_NAME,
            arguments=json.dumps(
                {
                    "mode": "create",
                    "steps": [
                        {
                            "id": "build",
                            "description": "Build the project structure",
                            "status": "in_progress",
                            "blocked_reason": None,
                        }
                    ],
                }
            ),
        ),
        ToolCall(id="mkdir", name="make_directory", arguments='{"path":"src"}'),
        ToolCall(
            id="write",
            name="write_file",
            arguments='{"path":"src/app.txt","content":"alpha"}',
        ),
        ToolCall(
            id="edit",
            name="edit_file",
            arguments=(
                '{"path":"src/app.txt","old_text":"alpha","new_text":"beta"}'
            ),
        ),
        ToolCall(id="read", name="read_file", arguments='{"path":"src/app.txt"}'),
    ]
    client = FakeClient(
        [
            ModelResponse(tool_calls=calls),
            ModelResponse(text="first incomplete final"),
            ModelResponse(text="done"),
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Build a tiny project.")

    assert result.status == "completed"
    assert state.plan is not None
    assert state.plan.revision == 1
    assert state.workspace_revision == 3
    assert result.verification_status == "unverified"
    assert result.plan_status == "active"
    observations = client.calls[1]["messages"][-5:]
    assert [message["tool_call_id"] for message in observations] == [
        "plan",
        "mkdir",
        "write",
        "edit",
        "read",
    ]
    assert all(message["role"] == "tool" for message in observations)


def test_partial_failure_increments_revision_and_still_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    original_mkdir = Path.mkdir

    def fail_app(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "app":
            raise OSError("simulated local failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_app)
    failed_call = ToolCall(
        id="failure",
        name="make_directory",
        arguments='{"path":"src/app"}',
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id=f"failure-{index}",
                        name=failed_call.name,
                        arguments=failed_call.arguments,
                    )
                ]
            )
            for index in range(3)
        ]
    )

    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path)),
    ).run("Create a directory despite a local failure.")

    assert result.status == "repeated_failure"
    assert state.workspace_revision == 1
    assert result.verification_status == "unverified"
    first_observation = json.loads(client.calls[1]["messages"][-1]["content"])
    assert first_observation["ok"] is False
    assert first_observation["data"]["workspace_changed"] is True
    assert first_observation["data"]["created_directories"] == ["src"]

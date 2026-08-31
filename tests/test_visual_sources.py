"""Stage 10C local visual-source and pending-payload tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from docx import Document
from rich.console import Console

import codeloop.agent.runner as runner_module
from codeloop.agent.events import ToolEvent
from codeloop.agent.requirements import UPDATE_REQUIREMENTS_ACTION_NAME
from codeloop.agent.runner import AgentRunner
from codeloop.agent.progress import (
    ProgressAction,
    ProgressFacts,
    ProgressState,
    ProgressTracker,
)
from codeloop.agent.task_state import TaskState
from codeloop.execution.tools import (
    MAX_PENDING_VISUALS,
    ToolRegistry,
)
from codeloop.execution.visual_sources import MAX_IMAGE_BYTES
from codeloop.execution.workspace import Workspace
from codeloop.interaction.console import ConsoleRenderer
from codeloop.model.client import ModelAPIError, ModelResponse, ToolCall


class FakeClient:
    supports_image_input = True

    def __init__(self, actions: list[ModelResponse | BaseException]) -> None:
        self._actions = iter(actions)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        action = next(self._actions)
        if isinstance(action, BaseException):
            raise action
        return action


def _image_bytes(image_type: str, size: int | None = None) -> bytes:
    signatures = {
        "png": b"\x89PNG\r\n\x1a\n",
        "jpeg": b"\xff\xd8\xff",
        "webp": b"RIFF\x00\x00\x00\x00WEBP",
    }
    raw = signatures[image_type]
    if size is None:
        return raw + b"payload"
    return raw + (b"x" * (size - len(raw)))


def _dispatch(registry: ToolRegistry, path: str) -> dict[str, object]:
    return registry.dispatch("read_image", json.dumps({"path": path}))


def _enabled_registry(root: Path) -> ToolRegistry:
    return ToolRegistry(Workspace(root), supports_image_input=True)


def test_supported_formats_return_role_neutral_descriptors(tmp_path: Path) -> None:
    fixtures = (
        ("1.png", "png", "image/png"),
        ("2.jpg", "jpeg", "image/jpeg"),
        ("3.JPEG", "jpeg", "image/jpeg"),
        ("4.webp", "webp", "image/webp"),
    )
    registry = _enabled_registry(tmp_path)

    original_bytes: dict[str, bytes] = {}
    for path, image_type, mime_type in fixtures:
        raw = _image_bytes(image_type)
        original_bytes[path] = raw
        (tmp_path / path).write_bytes(raw)
        result = _dispatch(registry, path)
        assert result == {
            "ok": True,
            "data": {
                "path": path,
                "image_type": image_type,
                "mime_type": mime_type,
                "size_bytes": len(raw),
            },
        }

    assert len(registry.snapshot_pending_visuals()) == MAX_PENDING_VISUALS
    serialized = repr(
        [payload.descriptor for payload in registry.snapshot_pending_visuals()]
    )
    assert "raw_bytes" not in serialized
    assert "base64" not in serialized
    assert "data:image" not in serialized
    assert {
        path: (tmp_path / path).read_bytes() for path, _type, _mime in fixtures
    } == original_bytes


def test_visual_tool_event_and_console_never_expose_payload(tmp_path: Path) -> None:
    raw = _image_bytes("png") + b"private-payload"
    (tmp_path / "1.png").write_bytes(raw)
    result = _dispatch(_enabled_registry(tmp_path), "1.png")
    event = ToolEvent(
        tool_call=ToolCall("image", "read_image", '{"path":"1.png"}'),
        result=result,
        dispatch_duration_ms=1,
        truncated=False,
    )
    rendered = Console(record=True, width=100)

    ConsoleRenderer(rendered).show_tool_event(event)

    public_event = repr(event)
    assert "private-payload" not in public_event
    assert "base64" not in public_event
    assert "data:image" not in public_event
    assert rendered.export_text() == ""


def test_capability_disabled_fails_before_adapter_access(tmp_path: Path) -> None:
    class UnexpectedAdapter:
        def load(self, _path: str):
            raise AssertionError("disabled vision must not access the adapter")

    registry = ToolRegistry(
        Workspace(tmp_path),
        supports_image_input=False,
        visual_source_adapter=UnexpectedAdapter(),  # type: ignore[arg-type]
    )
    result = _dispatch(registry, "missing.png")

    assert result["ok"] is False
    assert result["error_code"] == "vision_not_supported"
    assert registry.snapshot_pending_visuals() == ()


def test_disabled_capability_never_sends_image_to_model(tmp_path: Path) -> None:
    (tmp_path / "1.png").write_bytes(_image_bytes("png"))
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[ToolCall("image", "read_image", '{"path":"1.png"}')]
            ),
            ModelResponse(text="Image input is unavailable."),
        ]
    )
    result = AgentRunner(
        client,
        tools=ToolRegistry(Workspace(tmp_path), supports_image_input=False),
    ).run("Use 1.png")

    assert result.status == "completed"
    assert all(_image_urls(call) == [] for call in client.calls)
    observation = json.loads(client.calls[1]["messages"][-1]["content"])
    assert observation["error_code"] == "vision_not_supported"


def test_read_image_argument_contract(tmp_path: Path) -> None:
    registry = _enabled_registry(tmp_path)
    cases = (
        "{}",
        '{"path":""}',
        '{"path":true}',
        json.dumps({"path": "x" * 1001}),
        '{"path":"1.png","role":"requirement"}',
    )
    for arguments in cases:
        result = registry.dispatch("read_image", arguments)
        assert result["ok"] is False
        assert result["error_code"] == "invalid_arguments"
    assert registry.snapshot_pending_visuals() == ()


def test_type_signature_size_and_workspace_errors(tmp_path: Path) -> None:
    (tmp_path / "wrong.png").write_bytes(b"not png")
    (tmp_path / "notes.gif").write_bytes(b"GIF89a")
    exact = tmp_path / "exact.png"
    exact.write_bytes(_image_bytes("png", MAX_IMAGE_BYTES))
    oversized = tmp_path / "oversized.png"
    oversized.write_bytes(_image_bytes("png", MAX_IMAGE_BYTES + 1))
    registry = _enabled_registry(tmp_path)

    assert _dispatch(registry, "wrong.png")["error_code"] == "malformed_image"
    assert _dispatch(registry, "notes.gif")["error_code"] == "unsupported_image_type"
    assert _dispatch(registry, "missing.png")["error_code"] == "file_not_found"
    assert _dispatch(registry, "../outside.png")["error_code"] == "invalid_path"
    assert _dispatch(registry, str(exact))["error_code"] == "invalid_path"
    assert _dispatch(registry, "exact.png")["ok"] is True
    assert _dispatch(registry, "oversized.png")["error_code"] == "image_too_large"

    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    outside.write_bytes(_image_bytes("png"))
    escape = tmp_path / "escape.png"
    try:
        escape.symlink_to(outside)
    except (NotImplementedError, OSError):
        pass
    else:
        assert _dispatch(registry, "escape.png")["error_code"] == "invalid_path"


def test_pending_order_dedupe_replace_limits_and_cleanup(tmp_path: Path) -> None:
    for index in range(1, 6):
        (tmp_path / f"{index}.png").write_bytes(_image_bytes("png"))
    registry = _enabled_registry(tmp_path)
    for index in range(1, 5):
        assert _dispatch(registry, f"{index}.png")["ok"] is True

    original = registry.snapshot_pending_visuals()
    assert [item.descriptor.source_label for item in original] == [
        "1.png",
        "2.png",
        "3.png",
        "4.png",
    ]
    assert _dispatch(registry, "5.png")["error_code"] == "visual_attachment_limit"
    assert registry.snapshot_pending_visuals() == original

    replacement = _image_bytes("png") + b"replacement"
    (tmp_path / "2.png").write_bytes(replacement)
    assert _dispatch(registry, "2.png")["ok"] is True
    updated = registry.snapshot_pending_visuals()
    assert [item.descriptor.source_label for item in updated] == [
        "1.png",
        "2.png",
        "3.png",
        "4.png",
    ]
    assert updated[1].raw_bytes == replacement

    registry.consume_pending_visuals()
    registry.consume_pending_visuals()
    assert registry.snapshot_pending_visuals() == ()
    assert _dispatch(registry, "1.png")["ok"] is True
    registry.discard_pending_visuals()
    registry.discard_pending_visuals()
    assert registry.snapshot_pending_visuals() == ()


def test_total_pending_limit_is_atomic(tmp_path: Path) -> None:
    six_mib = 6 * 1024 * 1024
    for index in range(1, 4):
        (tmp_path / f"{index}.png").write_bytes(_image_bytes("png", six_mib))
    registry = _enabled_registry(tmp_path)

    assert _dispatch(registry, "1.png")["ok"] is True
    assert _dispatch(registry, "2.png")["ok"] is True
    before = registry.snapshot_pending_visuals()
    failed = _dispatch(registry, "3.png")
    assert failed["error_code"] == "visual_attachment_limit"
    assert registry.snapshot_pending_visuals() == before


def test_same_path_replacement_limit_failure_is_atomic(tmp_path: Path) -> None:
    sizes = {"1.png": 6, "2.png": 5, "3.png": 5}
    for path, size_mib in sizes.items():
        (tmp_path / path).write_bytes(
            _image_bytes("png", size_mib * 1024 * 1024)
        )
    registry = _enabled_registry(tmp_path)
    for path in sizes:
        assert _dispatch(registry, path)["ok"] is True
    before = registry.snapshot_pending_visuals()

    (tmp_path / "1.png").write_bytes(_image_bytes("png", MAX_IMAGE_BYTES))
    failed = _dispatch(registry, "1.png")
    assert failed["error_code"] == "visual_attachment_limit"
    assert registry.snapshot_pending_visuals() == before


def test_visual_eligibility_fifo_is_separate_and_not_snapshotted() -> None:
    state = TaskState()
    for index in range(18):
        state.record_execution_evidence(
            tool_name="read_image",
            result={"ok": True, "data": {"path": f"{index}.png"}},
        )
    assert state.read_visual_source_paths == tuple(
        f"{index}.png" for index in range(2, 18)
    )
    state.record_execution_evidence(
        tool_name="read_image",
        result={"ok": True, "data": {"path": "17.png"}},
    )
    assert state.read_visual_source_paths[-1] == "17.png"
    assert state.snapshot_for_model() is None
    assert TaskState().read_visual_source_paths == ()


def test_read_image_progress_uses_descriptor_only() -> None:
    result = {
        "ok": True,
        "data": {
            "path": "1.png",
            "image_type": "png",
            "mime_type": "image/png",
            "size_bytes": 12,
        },
    }
    action = ProgressAction(
        name="read_image",
        arguments='{"path":"1.png"}',
        result=result,
        workspace_revision=0,
        plan_before=None,
        plan_after=None,
    )
    tracker = ProgressTracker()
    state = ProgressState()
    facts = ProgressFacts(0, "not_required")

    assert tracker.evaluate_turn(state, before=facts, after=facts, actions=(action,)) == "continue"
    assert state.no_progress_turns == 0
    assert tracker.evaluate_turn(state, before=facts, after=facts, actions=(action,)) == "continue"
    assert state.no_progress_turns == 1


def test_read_image_is_eleventh_read_only_tool(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    assert len(registry.names) == 11
    assert registry.names[5] == "read_image"


def _visual_messages(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in call["messages"]
        if isinstance(message.get("content"), list)
    ]


def _image_urls(call: dict[str, Any]) -> list[str]:
    visual_messages = _visual_messages(call)
    if not visual_messages:
        return []
    return [
        item["image_url"]["url"]
        for item in visual_messages[-1]["content"]
        if item.get("type") == "image_url"
    ]


def _tool_messages(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in call["messages"]
        if message.get("role") == "tool"
    ]


def _requirements_arguments(paths: list[str]) -> str:
    return json.dumps(
        {
            "requirements": [
                {
                    "id": f"R{index}",
                    "kind": "functional" if index == 1 else "reference",
                    "description": (
                        "Support the visible requested operation"
                        if index == 1
                        else "Follow the visible layout hierarchy"
                    ),
                    "source": {"path": path, "locator": "visible region"},
                }
                for index, path in enumerate(paths, start=1)
            ]
        }
    )


def test_visual_only_turns_accumulate_then_requirements_consume_batch(
    tmp_path: Path,
) -> None:
    (tmp_path / "1.png").write_bytes(_image_bytes("png") + b"one")
    (tmp_path / "2.png").write_bytes(_image_bytes("png") + b"two")
    client = FakeClient(
        [
            ModelResponse(tool_calls=[ToolCall("one", "read_image", '{"path":"1.png"}')]),
            ModelResponse(
                text="I will inspect the second visual source.",
                tool_calls=[ToolCall("two", "read_image", '{"path":"2.png"}')],
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "requirements",
                        UPDATE_REQUIREMENTS_ACTION_NAME,
                        _requirements_arguments(["1.png", "2.png"]),
                    )
                ]
            ),
            ModelResponse(text="Requirements recorded."),
        ]
    )
    registry = _enabled_registry(tmp_path)
    result = AgentRunner(client, tools=registry).run("Build from 1.png and 2.png")

    assert result.status == "completed"
    assert result.workspace_revision == 0
    assert _image_urls(client.calls[0]) == []
    assert len(_image_urls(client.calls[1])) == 1
    assert len(_image_urls(client.calls[2])) == 2
    assert _image_urls(client.calls[3]) == []
    labels = [
        item["text"]
        for item in _visual_messages(client.calls[2])[-1]["content"]
        if item.get("type") == "text"
    ]
    assert labels == ["Visual source: 1.png", "Visual source: 2.png"]
    assert registry.snapshot_pending_visuals() == ()
    assert "data:image" not in repr(client.calls[3]["messages"])


@pytest.mark.parametrize(
    ("path", "kinds"),
    [
        ("1.png", ("functional",)),
        ("2.png", ("reference",)),
        ("3.png", ("functional", "reference")),
    ],
)
def test_neutral_image_names_support_content_driven_requirement_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    kinds: tuple[str, ...],
) -> None:
    (tmp_path / path).write_bytes(_image_bytes("png"))
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    requirements = json.dumps(
        {
            "requirements": [
                {
                    "id": f"R{index}",
                    "kind": kind,
                    "description": (
                        "Support the clearly visible operation"
                        if kind == "functional"
                        else "Follow the reliably visible layout"
                    ),
                    "source": {"path": path, "locator": "visible region"},
                }
                for index, kind in enumerate(kinds, start=1)
            ]
        }
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[ToolCall("image", "read_image", json.dumps({"path": path}))]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "requirements",
                        UPDATE_REQUIREMENTS_ACTION_NAME,
                        requirements,
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )

    result = AgentRunner(client, tools=_enabled_registry(tmp_path)).run(
        f"Build the project from {path} without pre-classifying its role"
    )

    assert result.status == "completed"
    assert len(_image_urls(client.calls[1])) == 1
    assert tuple(item.kind for item in state.requirements.requirements) == kinds
    assert {item.source.path for item in state.requirements.requirements} == {path}


def test_failed_visual_collection_keeps_previously_pending_image(
    tmp_path: Path,
) -> None:
    (tmp_path / "1.png").write_bytes(_image_bytes("png"))
    client = FakeClient(
        [
            ModelResponse(tool_calls=[ToolCall("one", "read_image", '{"path":"1.png"}')]),
            ModelResponse(tool_calls=[ToolCall("missing", "read_image", '{"path":"2.png"}')]),
            ModelResponse(text="The second visual source was unavailable."),
        ]
    )
    result = AgentRunner(client, tools=_enabled_registry(tmp_path)).run("Use images")

    assert result.status == "completed"
    assert len(_image_urls(client.calls[1])) == 1
    assert len(_image_urls(client.calls[2])) == 1
    failed_observation = json.loads(_tool_messages(client.calls[2])[-1]["content"])
    assert failed_observation["error_code"] == "file_not_found"


def test_mixed_read_image_turn_is_atomically_rejected_and_retains_batch(
    tmp_path: Path,
) -> None:
    for name in ("1.png", "2.png"):
        (tmp_path / name).write_bytes(_image_bytes("png") + name.encode())
    events: list[object] = []
    client = FakeClient(
        [
            ModelResponse(tool_calls=[ToolCall("one", "read_image", '{"path":"1.png"}')]),
            ModelResponse(
                tool_calls=[
                    ToolCall("two-mixed", "read_image", '{"path":"2.png"}'),
                    ToolCall(
                        "requirements-mixed",
                        UPDATE_REQUIREMENTS_ACTION_NAME,
                        _requirements_arguments(["1.png", "2.png"]),
                    ),
                ]
            ),
            ModelResponse(tool_calls=[ToolCall("two", "read_image", '{"path":"2.png"}')]),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "requirements",
                        UPDATE_REQUIREMENTS_ACTION_NAME,
                        _requirements_arguments(["1.png", "2.png"]),
                    )
                ]
            ),
            ModelResponse(text="done"),
        ]
    )
    result = AgentRunner(
        client,
        tools=_enabled_registry(tmp_path),
        on_tool_event=events.append,
    ).run("Use both images")

    assert result.status == "completed"
    mixed_results = _tool_messages(client.calls[2])[-2:]
    assert [message["tool_call_id"] for message in mixed_results] == [
        "two-mixed",
        "requirements-mixed",
    ]
    assert all(
        json.loads(message["content"])["error_code"] == "invalid_action_sequence"
        for message in mixed_results
    )
    assert len(_image_urls(client.calls[2])) == 1
    assert len(_image_urls(client.calls[3])) == 2
    assert [getattr(event, "tool_call").id for event in events] == ["one", "two"]


@pytest.mark.parametrize("other_name", ["update_plan", "list_files", "unknown"])
def test_mixed_visual_turn_rejects_core_execution_and_unknown_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    other_name: str,
) -> None:
    (tmp_path / "1.png").write_bytes(_image_bytes("png"))
    state = TaskState()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    events: list[object] = []
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall("image", "read_image", '{"path":"1.png"}'),
                    ToolCall("other", other_name, "{}"),
                ]
            ),
            ModelResponse(text="recovered"),
        ]
    )

    result = AgentRunner(
        client,
        tools=_enabled_registry(tmp_path),
        on_tool_event=events.append,
    ).run("task")

    assert result.status == "completed"
    observations = _tool_messages(client.calls[1])[-2:]
    assert [message["tool_call_id"] for message in observations] == [
        "image",
        "other",
    ]
    assert all(
        json.loads(message["content"])["error_code"]
        == "invalid_action_sequence"
        for message in observations
    )
    assert state.read_visual_source_paths == ()
    assert state.workspace_revision == 0
    assert events == []


def test_visual_retry_uses_frozen_payload_after_file_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "1.png"
    original = _image_bytes("png") + b"original"
    replacement = _image_bytes("png") + b"replacement"
    path.write_bytes(original)
    temporary = ModelAPIError(
        "temporary_api_error",
        "temporary",
        classification="retryable",
    )

    class MutatingClient(FakeClient):
        def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelResponse:
            self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
            call_number = len(self.calls)
            if call_number == 1:
                return ModelResponse(
                    tool_calls=[ToolCall("image", "read_image", '{"path":"1.png"}')]
                )
            if call_number == 2:
                path.write_bytes(replacement)
                raise temporary
            return ModelResponse(text="done")

    monkeypatch.setattr(runner_module, "sleep", lambda _delay: None)
    client = MutatingClient([])
    result = AgentRunner(client, tools=_enabled_registry(tmp_path)).run("Use 1.png")

    assert result.status == "completed"
    assert client.calls[1] == client.calls[2]
    assert path.read_bytes() == replacement


@pytest.mark.parametrize(
    ("actions", "max_steps", "expected_status"),
    [
        ([ModelResponse(text="done")], 3, "completed"),
        ([ModelResponse(tool_calls=[ToolCall("list", "list_files", "{}")])], 1, "max_steps"),
        ([KeyboardInterrupt()], 3, "user_interrupt"),
        ([RuntimeError("boom")], 3, "runtime_error"),
        (
            [
                ModelAPIError(
                    "model_api_error",
                    "rejected",
                    classification="fatal",
                )
            ],
            3,
            "fatal_api_error",
        ),
    ],
)
def test_pending_visuals_are_discarded_on_terminal_paths(
    tmp_path: Path,
    actions: list[ModelResponse | BaseException],
    max_steps: int,
    expected_status: str,
) -> None:
    (tmp_path / "1.png").write_bytes(_image_bytes("png"))
    registry = _enabled_registry(tmp_path)
    assert _dispatch(registry, "1.png")["ok"] is True
    discarded: list[bool] = []
    original_discard = registry.discard_pending_visuals

    def discard() -> None:
        discarded.append(True)
        original_discard()

    registry.discard_pending_visuals = discard  # type: ignore[method-assign]

    result = AgentRunner(FakeClient(actions), tools=registry, max_steps=max_steps).run("task")
    assert result.status == expected_status
    assert discarded == [True]
    assert registry.snapshot_pending_visuals() == ()


def test_payload_preparation_error_discards_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "1.png").write_bytes(_image_bytes("png"))
    registry = _enabled_registry(tmp_path)
    assert _dispatch(registry, "1.png")["ok"] is True
    monkeypatch.setattr(
        runner_module,
        "_visual_request_message",
        lambda _payloads: (_ for _ in ()).throw(RuntimeError("prepare")),
    )

    result = AgentRunner(FakeClient([]), tools=registry).run("task")
    assert result.status == "runtime_error"
    assert registry.snapshot_pending_visuals() == ()


def test_repeated_failure_no_progress_and_retry_exhaustion_all_discard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "sleep", lambda _delay: None)

    cases: list[tuple[list[ModelResponse | BaseException], int, str]] = [
        (
            [
                ModelResponse(tool_calls=[ToolCall(f"bad-{index}", "unknown", "{}")])
                for index in range(3)
            ],
            3,
            "repeated_failure",
        ),
        (
            [
                ModelResponse(tool_calls=[ToolCall(f"list-{index}", "list_files", "{}")])
                for index in range(7)
            ],
            10,
            "no_progress",
        ),
        (
            [
                ModelAPIError(
                    "temporary_api_error",
                    "temporary",
                    classification="retryable",
                )
                for _ in range(3)
            ],
            3,
            "fatal_api_error",
        ),
    ]
    for case_index, (actions, max_steps, expected) in enumerate(cases):
        image = tmp_path / f"case-{case_index}.png"
        image.write_bytes(_image_bytes("png"))
        registry = _enabled_registry(tmp_path)
        assert _dispatch(registry, image.name)["ok"] is True
        discarded: list[bool] = []
        original_discard = registry.discard_pending_visuals

        def discard(
            *,
            _discard=original_discard,
            _observed=discarded,
        ) -> None:
            _observed.append(True)
            _discard()

        registry.discard_pending_visuals = discard  # type: ignore[method-assign]
        result = AgentRunner(
            FakeClient(actions),
            tools=registry,
            max_steps=max_steps,
        ).run("task")
        assert result.status == expected
        assert discarded == [True]
        assert registry.snapshot_pending_visuals() == ()


def test_completion_review_and_unexpected_dispatch_discard_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "review.png").write_bytes(_image_bytes("png"))
    review_registry = _enabled_registry(tmp_path)
    assert _dispatch(review_registry, "review.png")["ok"] is True
    state = TaskState()
    state.record_workspace_change()
    monkeypatch.setattr(runner_module, "TaskState", lambda: state)
    reviewed = AgentRunner(
        FakeClient([ModelResponse(text="first"), ModelResponse(text="second")]),
        tools=review_registry,
    ).run("task")
    assert reviewed.status == "completed"
    assert review_registry.snapshot_pending_visuals() == ()

    (tmp_path / "dispatch.png").write_bytes(_image_bytes("png"))
    dispatch_registry = _enabled_registry(tmp_path)
    assert _dispatch(dispatch_registry, "dispatch.png")["ok"] is True
    definition = dispatch_registry._tools["list_files"]

    def broken_dispatch(_arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("dispatch")

    dispatch_registry._tools["list_files"] = type(definition)(
        definition.schema,
        broken_dispatch,
    )
    failed = AgentRunner(
        FakeClient(
            [ModelResponse(tool_calls=[ToolCall("list", "list_files", "{}")])]
        ),
        tools=dispatch_registry,
    ).run("task")
    assert failed.status == "runtime_error"
    assert dispatch_registry.snapshot_pending_visuals() == ()


def test_document_and_image_feed_one_requirement_state_and_runtime(
    tmp_path: Path,
) -> None:
    document = Document()
    document.add_paragraph("The page must support adding products.")
    document.save(tmp_path / "requirements.docx")
    (tmp_path / "1.png").write_bytes(_image_bytes("png") + b"layout")
    requirements = json.dumps(
        {
            "requirements": [
                {
                    "id": "D1",
                    "kind": "functional",
                    "description": "Users can add products",
                    "source": {
                        "path": "requirements.docx",
                        "locator": "paragraph 1",
                    },
                },
                {
                    "id": "V1",
                    "kind": "reference",
                    "description": "Follow the visible page layout",
                    "source": {"path": "1.png", "locator": "overall layout"},
                },
            ]
        }
    )
    client = FakeClient(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        "document",
                        "read_document",
                        '{"path":"requirements.docx"}',
                    )
                ]
            ),
            ModelResponse(
                tool_calls=[ToolCall("image", "read_image", '{"path":"1.png"}')]
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall("requirements", UPDATE_REQUIREMENTS_ACTION_NAME, requirements)
                ]
            ),
            ModelResponse(text="done"),
        ]
    )
    result = AgentRunner(client, tools=_enabled_registry(tmp_path)).run(
        "Use requirements.docx and 1.png"
    )

    assert result.status == "completed"
    assert len(_image_urls(client.calls[2])) == 1
    assert "The page must support adding products" in repr(client.calls[2]["messages"])
    assert _image_urls(client.calls[3]) == []

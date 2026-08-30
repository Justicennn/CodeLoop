"""Stage 9A deterministic repository-overview tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from codeloop.execution.tools import (
    MAX_OVERVIEW_DATA_CHARS,
    MAX_OVERVIEW_SCAN_ENTRIES,
    MAX_OVERVIEW_TREE_ENTRIES,
    ToolRegistry,
)
from codeloop.execution.workspace import Workspace


def _overview(
    root: Path,
    *,
    path: str = ".",
    max_depth: int = 3,
) -> dict[str, object]:
    result = ToolRegistry(Workspace(root)).dispatch(
        "repository_overview",
        json.dumps({"path": path, "max_depth": max_depth}),
    )
    assert result["ok"] is True
    return result["data"]


def test_empty_and_unicode_repository_are_relative_and_deterministic(
    tmp_path: Path,
) -> None:
    first = _overview(tmp_path)
    assert first["path"] == "."
    assert first["scan"]["scanned_entries"] == 0
    assert first["scan"]["complete"] is True
    assert first["tree"]["entries"] == []

    (tmp_path / "源码").mkdir()
    (tmp_path / "源码" / "模块.py").write_text("print('ok')", encoding="utf-8")
    second = _overview(tmp_path)
    third = _overview(tmp_path)
    assert second == third
    assert second["tree"]["entries"] == [
        {"path": "源码", "type": "directory", "depth": 1},
        {"path": "源码/模块.py", "type": "file", "depth": 2},
    ]


def test_tree_limit_does_not_stop_scan_or_late_anchor_detection(tmp_path: Path) -> None:
    for index in range(MAX_OVERVIEW_TREE_ENTRIES):
        (tmp_path / f"a{index:03d}.py").write_text("x", encoding="utf-8")
    late = tmp_path / "zz_late"
    late.mkdir()
    (late / "AGENTS.md").write_text("rules", encoding="utf-8")
    for index in range(20):
        (late / f"z{index:03d}.toml").write_text("x", encoding="utf-8")

    data = _overview(tmp_path)

    assert data["tree"]["count"] == MAX_OVERVIEW_TREE_ENTRIES
    assert data["tree"]["truncated"] is True
    assert "tree_entry_limit" in data["tree"]["truncation_reasons"]
    assert data["scan"]["complete"] is True
    assert data["scan"]["scanned_entries"] == MAX_OVERVIEW_TREE_ENTRIES + 22
    assert data["scan"]["scanned_files"] == MAX_OVERVIEW_TREE_ENTRIES + 21
    assert "zz_late/AGENTS.md" in data["anchors"]["items"]
    extensions = {
        item["extension"]: item["count"]
        for item in data["extension_stats"]["items"]
    }
    assert extensions[".py"] == MAX_OVERVIEW_TREE_ENTRIES
    assert extensions[".toml"] == 20


def test_depth_only_limits_tree_not_scan_or_statistics(tmp_path: Path) -> None:
    deep = tmp_path / "one" / "two" / "three"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("x", encoding="utf-8")

    data = _overview(tmp_path, max_depth=2)

    paths = [entry["path"] for entry in data["tree"]["entries"]]
    assert "one/two/three" not in paths
    assert "one/two/three/deep.py" not in paths
    assert data["scan"]["scanned_entries"] == 4
    assert data["scan"]["complete"] is True
    assert "tree_depth_limit" in data["truncation_reasons"]
    assert "tree_depth_limit" in data["tree"]["truncation_reasons"]
    assert data["extension_stats"]["items"] == [{"extension": ".py", "count": 1}]


def test_scan_limit_is_independent_and_reports_partial_counts(tmp_path: Path) -> None:
    for index in range(MAX_OVERVIEW_SCAN_ENTRIES + 1):
        (tmp_path / f"f{index:04d}.txt").touch()

    data = _overview(tmp_path)

    assert data["scan"]["scanned_entries"] == MAX_OVERVIEW_SCAN_ENTRIES
    assert data["scan"]["scanned_files"] == MAX_OVERVIEW_SCAN_ENTRIES
    assert data["scan"]["complete"] is False
    assert data["scan"]["truncated"] is True
    assert data["scan"]["truncation_reasons"] == ["scan_entry_limit"]
    assert "scan_entry_limit" in data["truncation_reasons"]


def test_final_serialized_data_hard_limit_includes_truncation_metadata(
    tmp_path: Path,
) -> None:
    for index in range(MAX_OVERVIEW_TREE_ENTRIES):
        # Keep the complete Windows temp path below the traditional MAX_PATH
        # boundary while still making 250 serialized tree entries exceed the
        # overview's 20,000-character output budget.
        long_name = f"{index:03d}_" + ("long_name_" * 8) + ".py"
        (tmp_path / long_name).touch()

    data = _overview(tmp_path)
    serialized = json.dumps(data, ensure_ascii=False, sort_keys=True)

    assert len(serialized) == data["serialized_chars"]
    assert len(serialized) <= MAX_OVERVIEW_DATA_CHARS
    assert "output_chars" in data["truncation_reasons"]
    assert data["tree"]["output_truncated"] is True
    assert "output_chars" in data["tree"]["truncation_reasons"]
    assert data["scan"]["scanned_files"] == MAX_OVERVIEW_TREE_ENTRIES
    assert data["scan"]["complete"] is True


def test_anchor_priority_and_lexical_tiebreak_are_fixed(tmp_path: Path) -> None:
    names = [
        ".gitignore",
        "Dockerfile.dev",
        "yarn.lock",
        "requirements-dev.txt",
        "package.json",
        "README-z.md",
        "README-a.md",
        "AGENTS.md",
    ]
    for name in names:
        (tmp_path / name).touch()

    data = _overview(tmp_path)

    assert data["anchors"]["items"] == [
        "AGENTS.md",
        "README-a.md",
        "README-z.md",
        "package.json",
        "requirements-dev.txt",
        "yarn.lock",
        "Dockerfile.dev",
        ".gitignore",
    ]


def test_anchor_section_limit_keeps_fixed_priority_prefix(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").touch()
    (tmp_path / ".gitignore").touch()
    for index in range(45):
        (tmp_path / f"README-{index:02d}.md").touch()

    data = _overview(tmp_path)
    anchors = data["anchors"]

    assert anchors["count"] == 40
    assert anchors["observed_count"] == 47
    assert anchors["items"][0] == "AGENTS.md"
    assert anchors["items"][1:] == [f"README-{index:02d}.md" for index in range(39)]
    assert ".gitignore" not in anchors["items"]
    assert anchors["truncation_reasons"] == ["anchor_limit"]


def test_ignored_directories_are_not_scanned(tmp_path: Path) -> None:
    generated = tmp_path / "node_modules"
    generated.mkdir()
    (generated / "package.json").touch()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").touch()

    data = _overview(tmp_path)

    assert data["scan"]["scanned_entries"] == 2
    assert all(
        not entry["path"].startswith("node_modules")
        for entry in data["tree"]["entries"]
    )


def test_directory_and_extension_sections_have_independent_caps(tmp_path: Path) -> None:
    for index in range(30):
        (tmp_path / f"module{index:02d}" / "src").mkdir(parents=True)
    for index in range(25):
        (tmp_path / f"file{index:02d}.ext{index:02d}").touch()

    data = _overview(tmp_path)

    directories = data["directory_candidates"]
    assert directories["observed_count"] == 30
    assert directories["count"] == 24
    assert directories["items"] == [
        f"module{index:02d}/src" for index in range(24)
    ]
    assert directories["truncation_reasons"] == ["directory_candidate_limit"]

    extensions = data["extension_stats"]
    assert extensions["observed_count"] == 25
    assert extensions["count"] == 20
    assert extensions["items"] == [
        {"extension": f".ext{index:02d}", "count": 1}
        for index in range(20)
    ]
    assert extensions["truncation_reasons"] == ["extension_stats_limit"]


def test_overview_rejects_escape_and_does_not_follow_symlink(tmp_path: Path) -> None:
    registry = ToolRegistry(Workspace(tmp_path))
    escaped = registry.dispatch(
        "repository_overview",
        json.dumps({"path": "../outside"}),
    )
    assert escaped["ok"] is False
    assert escaped["error_code"] == "invalid_path"

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    (outside / "secret.py").touch()
    link = tmp_path / "linked"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment")

    data = _overview(tmp_path)
    assert {entry["path"]: entry["type"] for entry in data["tree"]["entries"]}[
        "linked"
    ] == "symlink"
    assert data["scan"]["scanned_symlinks"] == 1
    assert all("secret.py" not in entry["path"] for entry in data["tree"]["entries"])

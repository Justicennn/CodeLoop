"""Workspace-bound coding tools and their direct dispatcher."""

from __future__ import annotations

import codecs
import json
import locale
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .command_policy import CommandApprovalRequest, dependency_mutation_request
from .document_sources import DocumentSourceError, extract_document
from .workspace import Workspace, WorkspaceError

ToolResult = dict[str, Any]
ToolHandler = Callable[[dict[str, Any]], ToolResult]

MAX_LIST_ITEMS = 200
MAX_LIST_DEPTH = 8
MAX_READ_LINES = 200
MAX_TEXT_CHARS = 20_000
MAX_DOCUMENT_CHARS = 20_000
MAX_SEARCH_MATCHES = 100
MAX_SEARCH_FILES = 50
MAX_OVERVIEW_PATH_CHARS = 1_000
MAX_OVERVIEW_DEPTH = 5
MAX_OVERVIEW_SCAN_ENTRIES = 5_000
MAX_OVERVIEW_TREE_ENTRIES = 250
MAX_OVERVIEW_ANCHORS = 40
MAX_OVERVIEW_DIRECTORY_CANDIDATES = 24
MAX_OVERVIEW_EXTENSION_STATS = 20
MAX_OVERVIEW_DATA_CHARS = 20_000
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
MAX_COMMAND_TIMEOUT_SECONDS = 300
TERMINATION_GRACE_SECONDS = 2

IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
}

_OVERVIEW_IGNORED_DIRECTORIES = IGNORED_DIRECTORIES | {"coverage", "target"}
_OVERVIEW_DIRECTORY_NAMES = {
    "src",
    "lib",
    "app",
    "tests",
    "test",
    "docs",
    "config",
}


@dataclass(frozen=True)
class ToolDefinition:
    schema: dict[str, Any]
    handler: ToolHandler
    managed_workspace_mutation: bool = False


class ToolInputError(Exception):
    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.data = data


def _success(data: dict[str, Any]) -> ToolResult:
    return {"ok": True, "data": data}


def _error(
    error_code: str,
    message: str,
    *,
    data: dict[str, Any] | None = None,
) -> ToolResult:
    result: ToolResult = {
        "ok": False,
        "error_code": error_code,
        "message": message,
    }
    if data is not None:
        result["data"] = data
    return result


def _validate_fields(
    arguments: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> None:
    missing = sorted(required - arguments.keys())
    if missing:
        raise ToolInputError(
            "invalid_arguments",
            f"Missing required field(s): {', '.join(missing)}",
        )
    unexpected = sorted(arguments.keys() - allowed)
    if unexpected:
        raise ToolInputError(
            "invalid_arguments",
            f"Unexpected field(s): {', '.join(unexpected)}",
        )


def _string_argument(
    arguments: dict[str, Any],
    name: str,
    *,
    default: str | None = None,
    allow_empty: bool = True,
) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or (not allow_empty and not value):
        requirement = "a non-empty string" if not allow_empty else "a string"
        raise ToolInputError("invalid_arguments", f"{name} must be {requirement}")
    return value


def _integer_argument(
    arguments: dict[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError("invalid_arguments", f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ToolInputError(
            "invalid_arguments",
            f"{name} must be between {minimum} and {maximum}",
        )
    return value


def _boolean_argument(
    arguments: dict[str, Any],
    name: str,
    *,
    default: bool,
) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ToolInputError("invalid_arguments", f"{name} must be a boolean")
    return value


def _decode_utf8(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolInputError(
            "unsupported_encoding",
            f"File is not valid UTF-8: {path.name}",
        ) from exc


def _local_command_output_encodings() -> tuple[str, ...]:
    """Return deterministic local-code-page fallbacks after UTF-8."""
    candidates = [
        *(("mbcs",) if os.name == "nt" else ()),
        locale.getencoding(),
        locale.getpreferredencoding(False),
    ]
    encodings: list[str] = []
    canonical_names: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            canonical = codecs.lookup(candidate).name
        except LookupError:
            continue
        if canonical == "utf-8" or canonical in canonical_names:
            continue
        canonical_names.add(canonical)
        encodings.append(candidate)
    return tuple(encodings)


def _decode_command_output(value: bytes) -> str:
    """Decode captured output as UTF-8 first, then the local code page.

    UTF-8-first avoids Windows locale decoding valid UTF-8 bytes into mojibake.
    The strict local attempts preserve output from legacy native programs, while
    the final replacement pass guarantees a string for malformed byte streams.
    """
    if not value:
        return ""
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        pass

    local_encodings = _local_command_output_encodings()
    for encoding in local_encodings:
        try:
            return value.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue

    if local_encodings:
        try:
            return value.decode(local_encodings[0], errors="replace")
        except LookupError:
            pass
    return value.decode("utf-8", errors="replace")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _detect_newline_style(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\n" in text:
        return "\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _logical_lines(text: str) -> list[str]:
    logical = _normalize_newlines(text)
    if not logical:
        return []
    lines = logical.split("\n")
    if logical.endswith("\n"):
        lines.pop()
    return lines


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _anchor_priority(filename: str) -> int | None:
    """Return the fixed overview retention tier for one file basename."""
    folded = filename.casefold()
    if folded == "agents.md":
        return 1
    if folded.startswith("readme"):
        return 2
    if (
        folded
        in {
            "pyproject.toml",
            "package.json",
            "cargo.toml",
            "go.mod",
            "pom.xml",
            "cmakelists.txt",
            "makefile",
        }
        or folded.startswith("build.gradle")
        or folded.startswith("settings.gradle")
        or (folded.startswith("requirements") and folded.endswith(".txt"))
    ):
        return 3
    if folded in {"package-lock.json", "pnpm-lock.yaml", "yarn.lock"}:
        return 4
    if folded.startswith("dockerfile") or folded.startswith("docker-compose"):
        return 5
    if folded == ".gitignore":
        return 6
    return None


LIST_FILES_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List a bounded portion of the workspace tree.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIST_DEPTH,
                    "default": 4,
                },
            },
            "additionalProperties": False,
        },
    },
}

REPOSITORY_OVERVIEW_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "repository_overview",
        "description": (
            "Return a deterministic, strictly bounded structural overview of a "
            "workspace directory. It does not read file contents or infer behavior."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "maxLength": MAX_OVERVIEW_PATH_CHARS,
                    "default": ".",
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_OVERVIEW_DEPTH,
                    "default": 3,
                },
            },
            "additionalProperties": False,
        },
    },
}

READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a bounded, line-numbered range from a UTF-8 text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1, "default": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

READ_DOCUMENT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_document",
        "description": (
            "Read a bounded deterministic text range from a local PDF or DOCX "
            "document. Continue with next_cursor when truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "cursor": {"type": "integer", "minimum": 0, "default": 0},
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_DOCUMENT_CHARS,
                    "default": MAX_DOCUMENT_CHARS,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

SEARCH_CODE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": "Search workspace text files using literal or regular-expression matching.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "path": {"type": "string", "default": "."},
                "regex": {"type": "boolean", "default": False},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

EDIT_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace exactly one matching text block in an existing UTF-8 file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
}

WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create a new UTF-8 text file without overwriting an existing path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}

MAKE_DIRECTORY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "make_directory",
        "description": (
            "Recursively create a directory inside the configured workspace root."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

RUN_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run one local program without a shell inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "cwd": {"type": "string", "default": "."},
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_COMMAND_TIMEOUT_SECONDS,
                    "default": DEFAULT_COMMAND_TIMEOUT_SECONDS,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


class ToolRegistry:
    """A direct registry of nine workspace-bound coding tools."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        sensitive_values: Iterable[str] = (),
    ) -> None:
        self._workspace = workspace
        self._sensitive_values = tuple(
            sorted(
                {value for value in sensitive_values if value},
                key=len,
                reverse=True,
            )
        )
        self._tools = {
            "repository_overview": ToolDefinition(
                REPOSITORY_OVERVIEW_SCHEMA,
                self._repository_overview,
            ),
            "list_files": ToolDefinition(LIST_FILES_SCHEMA, self._list_files),
            "read_file": ToolDefinition(READ_FILE_SCHEMA, self._read_file),
            "read_document": ToolDefinition(
                READ_DOCUMENT_SCHEMA,
                self._read_document,
            ),
            "search_code": ToolDefinition(SEARCH_CODE_SCHEMA, self._search_code),
            "edit_file": ToolDefinition(
                EDIT_FILE_SCHEMA,
                self._edit_file,
                managed_workspace_mutation=True,
            ),
            "write_file": ToolDefinition(
                WRITE_FILE_SCHEMA,
                self._write_file,
                managed_workspace_mutation=True,
            ),
            "make_directory": ToolDefinition(
                MAKE_DIRECTORY_SCHEMA,
                self._make_directory,
                managed_workspace_mutation=True,
            ),
            "run_command": ToolDefinition(RUN_COMMAND_SCHEMA, self._run_command),
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [definition.schema for definition in self._tools.values()]

    @property
    def names(self) -> tuple[str, ...]:
        """Expose registered names for narrow cross-layer collision checks."""
        return tuple(self._tools)

    def dispatch(self, name: str, arguments_json: str) -> ToolResult:
        definition = self._tools.get(name)
        if definition is None:
            return _error("unknown_tool", f"Unknown tool: {name}")

        try:
            arguments = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            result = _error(
                "invalid_arguments",
                "Tool arguments must be a JSON object",
            )
        else:
            if not isinstance(arguments, dict):
                result = _error(
                    "invalid_arguments",
                    "Tool arguments must be a JSON object",
                )
            else:
                try:
                    result = definition.handler(arguments)
                except (ToolInputError, WorkspaceError) as exc:
                    result = _error(exc.error_code, exc.message, data=exc.data)
                except OSError:
                    result = _error(
                        "tool_error",
                        "The local tool operation failed.",
                    )
        return self._normalize_mutation_result(definition, result)

    def command_approval_request(
        self,
        tool_name: str,
        arguments_json: str,
    ) -> CommandApprovalRequest | None:
        """Preflight one valid run_command without starting a subprocess."""
        if tool_name != "run_command":
            return None
        try:
            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict):
                return None
            command, _cwd, _timeout = self._validated_run_command_arguments(
                arguments
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ToolInputError,
            WorkspaceError,
        ):
            return None
        return dependency_mutation_request(command)

    def confirmed_workspace_change(
        self,
        tool_name: str,
        result: ToolResult,
    ) -> bool:
        """Trust effects only from registered managed-mutation tools."""
        definition = self._tools.get(tool_name)
        if definition is None or not definition.managed_workspace_mutation:
            return False
        data = result.get("data")
        return isinstance(data, dict) and data.get("workspace_changed") is True

    @staticmethod
    def _normalize_mutation_result(
        definition: ToolDefinition,
        result: ToolResult,
    ) -> ToolResult:
        if not definition.managed_workspace_mutation:
            return result
        normalized = dict(result)
        original_data = result.get("data")
        data = dict(original_data) if isinstance(original_data, dict) else {}
        if not isinstance(data.get("workspace_changed"), bool):
            data["workspace_changed"] = False
        normalized["data"] = data
        return normalized

    def _list_files(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(arguments, allowed={"path", "max_depth"})
        path_value = _string_argument(arguments, "path", default=".")
        max_depth = _integer_argument(
            arguments,
            "max_depth",
            default=4,
            minimum=1,
            maximum=MAX_LIST_DEPTH,
        )
        start = self._workspace.resolve(path_value, expected="directory")
        self._reject_ignored_path(start)
        entries: list[dict[str, str]] = []
        truncated = False

        def visit(directory: Path, depth: int) -> None:
            nonlocal truncated
            if truncated or depth > max_depth:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
            except OSError as exc:
                raise ToolInputError(
                    "file_access_error",
                    f"Cannot list directory: {self._workspace.relative_path(directory)}",
                ) from exc

            for child in children:
                if child.name in IGNORED_DIRECTORIES:
                    continue
                if len(entries) >= MAX_LIST_ITEMS:
                    truncated = True
                    return
                if child.is_symlink():
                    entry_type = "symlink"
                elif child.is_dir():
                    entry_type = "directory"
                else:
                    entry_type = "file"
                entries.append(
                    {
                        "path": self._workspace.relative_path(child),
                        "type": entry_type,
                    }
                )
                if entry_type == "directory" and depth < max_depth:
                    visit(child, depth + 1)
                    if truncated:
                        return

        visit(start, 1)
        return _success(
            {
                "path": self._workspace.relative_path(start),
                "entries": entries,
                "count": len(entries),
                "truncated": truncated,
            }
        )

    def _repository_overview(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(arguments, allowed={"path", "max_depth"})
        path_value = _string_argument(arguments, "path", default=".")
        if len(path_value) > MAX_OVERVIEW_PATH_CHARS:
            raise ToolInputError(
                "invalid_arguments",
                f"path must not exceed {MAX_OVERVIEW_PATH_CHARS} characters",
            )
        max_depth = _integer_argument(
            arguments,
            "max_depth",
            default=3,
            minimum=1,
            maximum=MAX_OVERVIEW_DEPTH,
        )
        start = self._workspace.resolve(path_value, expected="directory")
        start_relative = start.relative_to(self._workspace.root)
        if any(
            part in _OVERVIEW_IGNORED_DIRECTORIES
            for part in start_relative.parts
        ):
            raise ToolInputError(
                "invalid_path",
                "Repository overview cannot inspect ignored directories",
            )

        tree_entries: list[dict[str, Any]] = []
        anchor_candidates: list[tuple[int, str]] = []
        directory_candidates: list[str] = []
        extension_counts: dict[str, int] = {}
        scanned_entries = 0
        scanned_files = 0
        scanned_directories = 0
        scanned_symlinks = 0
        scan_truncated = False
        tree_depth_truncated = False
        tree_entry_truncated = False

        def visit(directory: Path, depth: int) -> bool:
            nonlocal scanned_entries
            nonlocal scanned_files
            nonlocal scanned_directories
            nonlocal scanned_symlinks
            nonlocal scan_truncated
            nonlocal tree_depth_truncated
            nonlocal tree_entry_truncated

            try:
                children = sorted(
                    directory.iterdir(),
                    key=lambda item: (item.name.casefold(), item.name),
                )
            except OSError as exc:
                raise ToolInputError(
                    "file_access_error",
                    "Cannot inspect directory: "
                    f"{self._workspace.relative_path(directory)}",
                ) from exc

            for child in children:
                if child.name in _OVERVIEW_IGNORED_DIRECTORIES:
                    continue
                if scanned_entries >= MAX_OVERVIEW_SCAN_ENTRIES:
                    scan_truncated = True
                    return False

                relative = self._workspace.relative_path(child)
                scanned_entries += 1
                if child.is_symlink():
                    entry_type = "symlink"
                    scanned_symlinks += 1
                elif child.is_dir():
                    entry_type = "directory"
                    scanned_directories += 1
                else:
                    entry_type = "file"
                    scanned_files += 1

                if depth <= max_depth:
                    if len(tree_entries) < MAX_OVERVIEW_TREE_ENTRIES:
                        tree_entries.append(
                            {"path": relative, "type": entry_type, "depth": depth}
                        )
                    else:
                        tree_entry_truncated = True
                else:
                    tree_depth_truncated = True

                if entry_type == "file":
                    priority = _anchor_priority(child.name)
                    if priority is not None:
                        anchor_candidates.append((priority, relative))
                    extension = child.suffix.casefold()
                    if extension:
                        extension_counts[extension] = (
                            extension_counts.get(extension, 0) + 1
                        )
                elif entry_type == "directory":
                    if child.name.casefold() in _OVERVIEW_DIRECTORY_NAMES:
                        directory_candidates.append(relative)
                    if not visit(child, depth + 1):
                        return False
            return True

        visit(start, 1)

        anchor_candidates.sort(
            key=lambda item: (item[0], item[1].casefold(), item[1])
        )
        all_anchors = [path for _, path in anchor_candidates]
        directory_candidates.sort(key=lambda path: (path.casefold(), path))
        extension_stats = [
            {"extension": extension, "count": count}
            for extension, count in sorted(
                extension_counts.items(),
                key=lambda item: (-item[1], item[0].casefold(), item[0]),
            )
        ]

        anchors_limited = len(all_anchors) > MAX_OVERVIEW_ANCHORS
        directories_limited = (
            len(directory_candidates) > MAX_OVERVIEW_DIRECTORY_CANDIDATES
        )
        extensions_limited = len(extension_stats) > MAX_OVERVIEW_EXTENSION_STATS
        anchors = all_anchors[:MAX_OVERVIEW_ANCHORS]
        directories = directory_candidates[:MAX_OVERVIEW_DIRECTORY_CANDIDATES]
        extensions = extension_stats[:MAX_OVERVIEW_EXTENSION_STATS]
        output_truncated = {
            "tree": False,
            "anchors": False,
            "directory_candidates": False,
            "extension_stats": False,
        }
        reasons: set[str] = set()
        if scan_truncated:
            reasons.add("scan_entry_limit")
        if tree_depth_truncated:
            reasons.add("tree_depth_limit")
        if tree_entry_truncated:
            reasons.add("tree_entry_limit")
        if anchors_limited:
            reasons.add("anchor_limit")
        if directories_limited:
            reasons.add("directory_candidate_limit")
        if extensions_limited:
            reasons.add("extension_stats_limit")

        def build_data() -> dict[str, Any]:
            scan_reasons = ["scan_entry_limit"] if scan_truncated else []
            tree_reasons = []
            if tree_depth_truncated:
                tree_reasons.append("tree_depth_limit")
            if tree_entry_truncated:
                tree_reasons.append("tree_entry_limit")
            if output_truncated["tree"]:
                tree_reasons.append("output_chars")
            anchor_reasons = []
            if anchors_limited:
                anchor_reasons.append("anchor_limit")
            if output_truncated["anchors"]:
                anchor_reasons.append("output_chars")
            directory_reasons = []
            if directories_limited:
                directory_reasons.append("directory_candidate_limit")
            if output_truncated["directory_candidates"]:
                directory_reasons.append("output_chars")
            extension_reasons = []
            if extensions_limited:
                extension_reasons.append("extension_stats_limit")
            if output_truncated["extension_stats"]:
                extension_reasons.append("output_chars")
            data: dict[str, Any] = {
                "path": self._workspace.relative_path(start),
                "scan": {
                    "entry_limit": MAX_OVERVIEW_SCAN_ENTRIES,
                    "scanned_entries": scanned_entries,
                    "scanned_files": scanned_files,
                    "scanned_directories": scanned_directories,
                    "scanned_symlinks": scanned_symlinks,
                    "complete": not scan_truncated,
                    "truncated": scan_truncated,
                    "truncation_reasons": scan_reasons,
                },
                "tree": {
                    "max_depth": max_depth,
                    "entry_limit": MAX_OVERVIEW_TREE_ENTRIES,
                    "entries": list(tree_entries),
                    "count": len(tree_entries),
                    "truncated": (
                        tree_depth_truncated
                        or tree_entry_truncated
                        or output_truncated["tree"]
                    ),
                    "output_truncated": output_truncated["tree"],
                    "truncation_reasons": tree_reasons,
                },
                "anchors": {
                    "limit": MAX_OVERVIEW_ANCHORS,
                    "observed_count": len(all_anchors),
                    "items": list(anchors),
                    "count": len(anchors),
                    "truncated": anchors_limited or output_truncated["anchors"],
                    "output_truncated": output_truncated["anchors"],
                    "truncation_reasons": anchor_reasons,
                },
                "directory_candidates": {
                    "limit": MAX_OVERVIEW_DIRECTORY_CANDIDATES,
                    "observed_count": len(directory_candidates),
                    "items": list(directories),
                    "count": len(directories),
                    "truncated": (
                        directories_limited
                        or output_truncated["directory_candidates"]
                    ),
                    "output_truncated": output_truncated["directory_candidates"],
                    "truncation_reasons": directory_reasons,
                },
                "extension_stats": {
                    "limit": MAX_OVERVIEW_EXTENSION_STATS,
                    "observed_count": len(extension_stats),
                    "items": list(extensions),
                    "count": len(extensions),
                    "truncated": (
                        extensions_limited or output_truncated["extension_stats"]
                    ),
                    "output_truncated": output_truncated["extension_stats"],
                    "truncation_reasons": extension_reasons,
                },
                "data_char_limit": MAX_OVERVIEW_DATA_CHARS,
                "serialized_chars": 0,
                "truncated": bool(reasons),
                "truncation_reasons": sorted(reasons),
            }
            while True:
                measured = len(
                    json.dumps(data, ensure_ascii=False, sort_keys=True)
                )
                if data["serialized_chars"] == measured:
                    return data
                data["serialized_chars"] = measured

        while True:
            data = build_data()
            if data["serialized_chars"] <= MAX_OVERVIEW_DATA_CHARS:
                return _success(data)
            reasons.add("output_chars")
            if tree_entries:
                tree_entries.pop()
                output_truncated["tree"] = True
            elif directories:
                directories.pop()
                output_truncated["directory_candidates"] = True
            elif extensions:
                extensions.pop()
                output_truncated["extension_stats"] = True
            elif anchors:
                anchors.pop()
                output_truncated["anchors"] = True
            else:
                raise ToolInputError(
                    "tool_error",
                    "Repository overview metadata exceeds its output limit.",
                )

    def _read_file(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(
            arguments,
            allowed={"path", "start_line", "end_line"},
            required={"path"},
        )
        path_value = _string_argument(arguments, "path")
        start_line = _integer_argument(
            arguments,
            "start_line",
            default=1,
            minimum=1,
            maximum=2_147_483_647,
        )
        end_value = arguments.get("end_line")
        if end_value is not None:
            if isinstance(end_value, bool) or not isinstance(end_value, int):
                raise ToolInputError("invalid_arguments", "end_line must be an integer or null")
            if end_value < start_line:
                raise ToolInputError(
                    "invalid_arguments",
                    "end_line must be greater than or equal to start_line",
                )

        path = self._workspace.resolve(path_value, expected="file")
        content = _decode_utf8(path)
        lines = _logical_lines(content)
        total_lines = len(lines)
        requested_end = end_value if end_value is not None else total_lines
        limited_end = min(requested_end, start_line + MAX_READ_LINES - 1, total_lines)

        numbered = "\n".join(
            f"{number}: {lines[number - 1]}"
            for number in range(start_line, limited_end + 1)
            if number <= total_lines
        )
        numbered, chars_truncated = _truncate(numbered)
        lines_truncated = (
            total_lines >= start_line
            and requested_end > start_line + MAX_READ_LINES - 1
        )
        return _success(
            {
                "path": self._workspace.relative_path(path),
                "start_line": start_line,
                "end_line": limited_end,
                "total_lines": total_lines,
                "content": numbered,
                "truncated": chars_truncated or lines_truncated,
            }
        )

    def _read_document(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(
            arguments,
            allowed={"path", "cursor", "max_chars"},
            required={"path"},
        )
        path_value = _string_argument(arguments, "path", allow_empty=False)
        cursor = _integer_argument(
            arguments,
            "cursor",
            default=0,
            minimum=0,
            maximum=2_147_483_647,
        )
        max_chars = _integer_argument(
            arguments,
            "max_chars",
            default=MAX_DOCUMENT_CHARS,
            minimum=1,
            maximum=MAX_DOCUMENT_CHARS,
        )
        path = self._workspace.resolve(path_value, expected="file")
        try:
            document = extract_document(path)
        except DocumentSourceError as exc:
            raise ToolInputError(exc.error_code, exc.message) from exc

        total_chars = len(document.text)
        if cursor > total_chars:
            raise ToolInputError(
                "invalid_arguments",
                "cursor cannot exceed the document's total_chars.",
                data={"cursor": cursor, "total_chars": total_chars},
            )
        end_cursor = min(total_chars, cursor + max_chars)
        truncated = end_cursor < total_chars
        first_unit = document.locator_at(cursor) if cursor < end_cursor else None
        last_unit = (
            document.locator_at(end_cursor - 1) if cursor < end_cursor else None
        )
        return _success(
            {
                "path": self._workspace.relative_path(path),
                "document_type": document.document_type,
                "text": document.text[cursor:end_cursor],
                "position": {
                    "start_cursor": cursor,
                    "end_cursor": end_cursor,
                    "total_chars": total_chars,
                    "first_unit": first_unit,
                    "last_unit": last_unit,
                },
                "truncated": truncated,
                "next_cursor": end_cursor if truncated else None,
            }
        )

    def _search_code(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(
            arguments,
            allowed={"query", "path", "regex"},
            required={"query"},
        )
        query = _string_argument(arguments, "query", allow_empty=False)
        path_value = _string_argument(arguments, "path", default=".")
        use_regex = _boolean_argument(arguments, "regex", default=False)
        search_root = self._workspace.resolve(path_value)
        self._reject_ignored_path(search_root)

        compiled: re.Pattern[str] | None = None
        if use_regex:
            try:
                compiled = re.compile(query)
            except re.error as exc:
                raise ToolInputError(
                    "invalid_arguments",
                    f"Invalid regular expression: {exc}",
                ) from exc

        rg_path = shutil.which("rg")
        if rg_path is not None:
            return self._search_with_rg(rg_path, search_root, query, use_regex)
        return self._search_with_python(search_root, query, compiled)

    def _reject_ignored_path(self, path: Path) -> None:
        relative = path.relative_to(self._workspace.root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            raise ToolInputError(
                "invalid_path",
                "Listing or searching ignored directories is not allowed",
            )

    def _search_with_rg(
        self,
        rg_path: str,
        search_root: Path,
        query: str,
        use_regex: bool,
    ) -> ToolResult:
        command = [rg_path, "--json", "--line-number", "--color", "never"]
        if not use_regex:
            command.append("--fixed-strings")
        for ignored in sorted(IGNORED_DIRECTORIES):
            command.extend(["--glob", f"!**/{ignored}/**"])
        command.extend(["--", query, str(search_root)])

        completed = subprocess.run(
            command,
            cwd=self._workspace.root,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode not in (0, 1):
            raise ToolInputError(
                "search_error",
                completed.stderr.strip() or "ripgrep search failed",
            )

        matches: list[dict[str, Any]] = []
        matched_files: set[str] = set()
        used_chars = 0
        truncated = False
        for raw_line in completed.stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event["data"]
            path = Path(data["path"]["text"]).resolve(strict=True)
            relative = self._workspace.relative_path(path)
            text = data["lines"]["text"].rstrip("\r\n")
            if relative not in matched_files and len(matched_files) >= MAX_SEARCH_FILES:
                truncated = True
                break
            if len(matches) >= MAX_SEARCH_MATCHES or used_chars + len(text) > MAX_TEXT_CHARS:
                truncated = True
                break
            matched_files.add(relative)
            used_chars += len(text)
            matches.append(
                {
                    "path": relative,
                    "line": data["line_number"],
                    "text": text,
                }
            )
        return _success(
            {
                "query": query,
                "matches": matches,
                "count": len(matches),
                "files": len(matched_files),
                "truncated": truncated,
                "engine": "rg",
            }
        )

    def _search_with_python(
        self,
        search_root: Path,
        query: str,
        compiled: re.Pattern[str] | None,
    ) -> ToolResult:
        matches: list[dict[str, Any]] = []
        matched_files: set[str] = set()
        used_chars = 0
        truncated = False

        for path in self._search_files(search_root):
            try:
                content = path.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            relative = self._workspace.relative_path(path)
            for line_number, line in enumerate(content.splitlines(), start=1):
                is_match = compiled.search(line) is not None if compiled else query in line
                if not is_match:
                    continue
                if relative not in matched_files and len(matched_files) >= MAX_SEARCH_FILES:
                    truncated = True
                    break
                if len(matches) >= MAX_SEARCH_MATCHES or used_chars + len(line) > MAX_TEXT_CHARS:
                    truncated = True
                    break
                matched_files.add(relative)
                used_chars += len(line)
                matches.append({"path": relative, "line": line_number, "text": line})
            if truncated:
                break

        return _success(
            {
                "query": query,
                "matches": matches,
                "count": len(matches),
                "files": len(matched_files),
                "truncated": truncated,
                "engine": "python",
            }
        )

    def _search_files(self, search_root: Path) -> Iterator[Path]:
        if search_root.is_file():
            yield search_root
            return
        for current, directories, files in os.walk(search_root, followlinks=False):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in IGNORED_DIRECTORIES
                and not (Path(current) / directory).is_symlink()
            )
            for filename in sorted(files):
                path = Path(current) / filename
                if not path.is_symlink():
                    yield path

    def _edit_file(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(
            arguments,
            allowed={"path", "old_text", "new_text"},
            required={"path", "old_text", "new_text"},
        )
        path_value = _string_argument(arguments, "path")
        old_text = _string_argument(arguments, "old_text", allow_empty=False)
        new_text = _string_argument(arguments, "new_text")
        path = self._workspace.resolve(path_value, expected="file")
        content = _decode_utf8(path)
        newline_style = _detect_newline_style(content)
        logical_content = _normalize_newlines(content)
        logical_old_text = _normalize_newlines(old_text)
        logical_new_text = _normalize_newlines(new_text)
        match_count = logical_content.count(logical_old_text)

        if match_count == 0:
            return _error(
                "edit_mismatch",
                "old_text was not found",
                data={"matches": 0},
            )
        if match_count > 1:
            return _error(
                "edit_ambiguous",
                "old_text must match exactly once",
                data={"matches": match_count},
            )

        logical_updated = logical_content.replace(
            logical_old_text,
            logical_new_text,
            1,
        )
        updated = logical_updated.replace("\n", newline_style)
        if updated == content:
            return _success(
                {
                    "path": self._workspace.relative_path(path),
                    "replacements": 1,
                    "before_chars": len(content),
                    "after_chars": len(updated),
                    "workspace_changed": False,
                }
            )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            ) as temporary:
                temporary.write(updated.encode("utf-8"))
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            revalidated = self._workspace.resolve(path_value, expected="file")
            if revalidated != path:
                raise WorkspaceError(
                    "invalid_path",
                    "The edit target changed during execution.",
                )
            os.replace(temporary_path, revalidated)
            path = revalidated
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        return _success(
            {
                "path": self._workspace.relative_path(path),
                "replacements": 1,
                "before_chars": len(content),
                "after_chars": len(updated),
                "workspace_changed": True,
            }
        )

    def _write_file(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(
            arguments,
            allowed={"path", "content"},
            required={"path", "content"},
        )
        path_value = _string_argument(arguments, "path")
        content = _string_argument(arguments, "content")
        path = self._workspace.resolve_new_file(path_value)
        created = False
        try:
            output = path.open("xb")
            created = True
            with output:
                output.write(content.encode("utf-8"))
        except FileExistsError:
            return _error("file_exists", f"File already exists: {path_value}")
        except OSError:
            return _error(
                "tool_error",
                "The local file operation failed.",
                data={
                    "path": path_value,
                    "workspace_changed": created and path.exists(),
                },
            )
        return _success(
            {
                "path": self._workspace.relative_path(path),
                "characters": len(content),
                "workspace_changed": True,
            }
        )

    def _make_directory(self, arguments: dict[str, Any]) -> ToolResult:
        _validate_fields(arguments, allowed={"path"}, required={"path"})
        path_value = _string_argument(arguments, "path", allow_empty=False)
        target = self._workspace.resolve_directory_target(path_value)
        target_relative = self._workspace.relative_path(target)

        if target.exists():
            if not target.is_dir():
                return _error(
                    "path_conflict",
                    f"Directory target conflicts with a file: {path_value}",
                )
            return _success(
                {
                    "path": target_relative,
                    "created_directories": [],
                    "created_count": 0,
                    "workspace_changed": False,
                }
            )

        missing: list[Path] = []
        current = target
        while not current.exists():
            missing.append(current)
            current = current.parent

        if not current.is_dir():
            return _error(
                "path_conflict",
                f"A parent path is not a directory: {path_value}",
            )

        created_directories: list[str] = []
        for directory in reversed(missing):
            relative = self._workspace.relative_path(directory)
            try:
                parent_relative = self._workspace.relative_path(directory.parent)
                self._workspace.resolve(parent_relative, expected="directory")
                directory.mkdir()
                created_directories.append(relative)
                self._workspace.resolve(relative, expected="directory")
            except FileExistsError:
                if directory.is_dir():
                    continue
                return self._directory_failure(
                    "path_conflict",
                    f"Directory target conflicts with a file: {path_value}",
                    path=target_relative,
                    created_directories=created_directories,
                )
            except WorkspaceError as exc:
                return self._directory_failure(
                    exc.error_code,
                    exc.message,
                    path=target_relative,
                    created_directories=created_directories,
                )
            except OSError:
                return self._directory_failure(
                    "tool_error",
                    "The local directory operation failed.",
                    path=target_relative,
                    created_directories=created_directories,
                )

        try:
            resolved = self._workspace.resolve(path_value, expected="directory")
        except WorkspaceError as exc:
            return self._directory_failure(
                exc.error_code,
                exc.message,
                path=target_relative,
                created_directories=created_directories,
            )
        return _success(
            {
                "path": self._workspace.relative_path(resolved),
                "created_directories": created_directories,
                "created_count": len(created_directories),
                "workspace_changed": bool(created_directories),
            }
        )

    @staticmethod
    def _directory_failure(
        error_code: str,
        message: str,
        *,
        path: str,
        created_directories: list[str],
    ) -> ToolResult:
        return _error(
            error_code,
            message,
            data={
                "path": path,
                "created_directories": list(created_directories),
                "created_count": len(created_directories),
                "workspace_changed": bool(created_directories),
            },
        )

    def _run_command(self, arguments: dict[str, Any]) -> ToolResult:
        command, cwd, timeout_seconds = self._validated_run_command_arguments(
            arguments
        )

        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            return _error(
                "command_not_found",
                "The requested executable was not found.",
            )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stdout_bytes, stderr_bytes = self._terminate_and_collect(process)
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            data = self._command_result_data(
                command=command,
                cwd=cwd,
                exit_code=process.returncode,
                stdout=_decode_command_output(stdout_bytes),
                stderr=_decode_command_output(stderr_bytes),
                duration_ms=duration_ms,
                timeout_seconds=timeout_seconds,
                timed_out=True,
                direct_child_reaped=process.poll() is not None,
            )
            return _error(
                "command_timeout",
                f"Command exceeded the {timeout_seconds} second timeout.",
                data=data,
            )
        except KeyboardInterrupt:
            self._terminate_and_collect(process)
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        data = self._command_result_data(
            command=command,
            cwd=cwd,
            exit_code=process.returncode,
            stdout=_decode_command_output(stdout),
            stderr=_decode_command_output(stderr),
            duration_ms=duration_ms,
            timeout_seconds=timeout_seconds,
            timed_out=False,
            direct_child_reaped=True,
        )
        if process.returncode != 0:
            return _error(
                "command_failed",
                f"Command exited with code {process.returncode}",
                data=data,
            )
        return _success(data)

    def _validated_run_command_arguments(
        self,
        arguments: dict[str, Any],
    ) -> tuple[list[str], Path, int]:
        _validate_fields(
            arguments,
            allowed={"command", "cwd", "timeout_seconds"},
            required={"command"},
        )
        command = arguments.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or "\x00" in item for item in command)
            or not command[0]
        ):
            raise ToolInputError(
                "invalid_arguments",
                "command must be a non-empty array of strings",
            )
        cwd_value = _string_argument(arguments, "cwd", default=".")
        timeout_seconds = _integer_argument(
            arguments,
            "timeout_seconds",
            default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            minimum=1,
            maximum=MAX_COMMAND_TIMEOUT_SECONDS,
        )
        cwd = self._workspace.resolve(cwd_value, expected="directory")
        return list(command), cwd, timeout_seconds

    def _terminate_and_collect(
        self,
        process: subprocess.Popen[bytes],
    ) -> tuple[bytes, bytes]:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        try:
            stdout, stderr = process.communicate(
                timeout=TERMINATION_GRACE_SECONDS
            )
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            stdout, stderr = process.communicate()
        return stdout or b"", stderr or b""

    def _command_result_data(
        self,
        *,
        command: list[str],
        cwd: Path,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        duration_ms: float,
        timeout_seconds: int,
        timed_out: bool,
        direct_child_reaped: bool,
    ) -> dict[str, Any]:
        safe_command = [self._redact(value) for value in command]
        stdout, stdout_truncated = _truncate(self._redact(stdout))
        stderr, stderr_truncated = _truncate(self._redact(stderr))
        data = {
            "command": safe_command,
            "cwd": self._workspace.relative_path(cwd),
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "duration_ms": duration_ms,
            "timeout_seconds": timeout_seconds,
            "timed_out": timed_out,
            "direct_child_reaped": direct_child_reaped,
        }
        return data

    def _redact(self, value: str) -> str:
        redacted = value
        for sensitive in self._sensitive_values:
            redacted = redacted.replace(sensitive, "[REDACTED]")
        return redacted

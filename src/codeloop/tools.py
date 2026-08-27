"""Workspace-bound coding tools and their direct dispatcher."""

from __future__ import annotations

import json
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

from .workspace import Workspace, WorkspaceError

ToolResult = dict[str, Any]
ToolHandler = Callable[[dict[str, Any]], ToolResult]

MAX_LIST_ITEMS = 200
MAX_LIST_DEPTH = 8
MAX_READ_LINES = 200
MAX_TEXT_CHARS = 20_000
MAX_SEARCH_MATCHES = 100
MAX_SEARCH_FILES = 50
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


@dataclass(frozen=True)
class ToolDefinition:
    schema: dict[str, Any]
    handler: ToolHandler


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
    """A direct registry of six workspace-bound coding tools."""

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
            "list_files": ToolDefinition(LIST_FILES_SCHEMA, self._list_files),
            "read_file": ToolDefinition(READ_FILE_SCHEMA, self._read_file),
            "search_code": ToolDefinition(SEARCH_CODE_SCHEMA, self._search_code),
            "edit_file": ToolDefinition(EDIT_FILE_SCHEMA, self._edit_file),
            "write_file": ToolDefinition(WRITE_FILE_SCHEMA, self._write_file),
            "run_command": ToolDefinition(RUN_COMMAND_SCHEMA, self._run_command),
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [definition.schema for definition in self._tools.values()]

    def dispatch(self, name: str, arguments_json: str) -> ToolResult:
        definition = self._tools.get(name)
        if definition is None:
            return _error("unknown_tool", f"Unknown tool: {name}")

        try:
            arguments = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            return _error("invalid_arguments", "Tool arguments must be a JSON object")
        if not isinstance(arguments, dict):
            return _error("invalid_arguments", "Tool arguments must be a JSON object")

        try:
            return definition.handler(arguments)
        except (ToolInputError, WorkspaceError) as exc:
            return _error(exc.error_code, exc.message, data=exc.data)
        except OSError:
            return _error("tool_error", "The local tool operation failed.")

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
        try:
            with path.open("xb") as output:
                output.write(content.encode("utf-8"))
        except FileExistsError:
            return _error("file_exists", f"File already exists: {path_value}")
        return _success(
            {
                "path": self._workspace.relative_path(path),
                "characters": len(content),
            }
        )

    def _run_command(self, arguments: dict[str, Any]) -> ToolResult:
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

        started = time.perf_counter()
        try:
            cwd = self._workspace.resolve(cwd_value, expected="directory")
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
        except FileNotFoundError:
            return _error(
                "command_not_found",
                "The requested executable was not found.",
            )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stdout, stderr = self._terminate_and_collect(process)
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            data = self._command_result_data(
                command=command,
                cwd=cwd,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
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
            stdout=stdout,
            stderr=stderr,
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

    def _terminate_and_collect(
        self,
        process: subprocess.Popen[str],
    ) -> tuple[str, str]:
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
        return stdout or "", stderr or ""

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

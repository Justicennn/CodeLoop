"""Workspace-bound path validation for local tool execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WorkspaceError(Exception):
    """A safe, structured workspace validation failure."""

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


class Workspace:
    """Resolve model-provided paths without allowing workspace escape."""

    def __init__(self, root: str | Path) -> None:
        try:
            resolved = Path(root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError(
                "invalid_workspace",
                f"Workspace does not exist or cannot be resolved: {root}",
            ) from exc
        if not resolved.is_dir():
            raise WorkspaceError("invalid_workspace", "Workspace must be a directory")
        self._root = resolved

    @property
    def root(self) -> Path:
        return self._root

    def resolve(
        self,
        value: str,
        *,
        must_exist: bool = True,
        expected: str | None = None,
    ) -> Path:
        relative = self._validate_relative(value)
        try:
            candidate = (self._root / relative).resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise WorkspaceError("file_not_found", f"Path does not exist: {value}") from exc
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("invalid_path", f"Path cannot be resolved: {value}") from exc

        if candidate != self._root and self._root not in candidate.parents:
            raise WorkspaceError("invalid_path", f"Path escapes workspace: {value}")
        if must_exist and expected == "file" and not candidate.is_file():
            raise WorkspaceError("invalid_path", f"Expected a file: {value}")
        if must_exist and expected == "directory" and not candidate.is_dir():
            raise WorkspaceError("invalid_path", f"Expected a directory: {value}")
        return candidate

    def resolve_new_file(self, value: str) -> Path:
        candidate = self.resolve(value, must_exist=False)
        try:
            parent = candidate.parent.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise WorkspaceError(
                "directory_not_found",
                f"Parent directory does not exist: {value}",
            ) from exc
        if parent != self._root and self._root not in parent.parents:
            raise WorkspaceError("invalid_path", f"Path escapes workspace: {value}")
        if not parent.is_dir():
            raise WorkspaceError(
                "directory_not_found",
                f"Parent path is not a directory: {value}",
            )
        return candidate

    def resolve_directory_target(self, value: str) -> Path:
        """Resolve an existing or new directory target inside the fixed root.

        Existing path components are resolved strictly so symlinks cannot move
        the target outside the workspace.  Missing components are returned as a
        canonical tail below the nearest existing workspace directory; this
        method never creates the workspace root or any child directory.
        """
        relative = self._validate_relative(value)
        current = self._root
        parts = relative.parts

        for index, part in enumerate(parts):
            candidate = current / part
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError:
                try:
                    target = current.joinpath(*parts[index:]).resolve(strict=False)
                except (OSError, RuntimeError) as exc:
                    raise WorkspaceError(
                        "invalid_path",
                        f"Path cannot be resolved: {value}",
                    ) from exc
                self._ensure_contained(target, value)
                return target
            except (OSError, RuntimeError) as exc:
                raise WorkspaceError(
                    "invalid_path",
                    f"Path cannot be resolved: {value}",
                ) from exc

            self._ensure_contained(resolved, value)
            if not resolved.is_dir():
                raise WorkspaceError(
                    "path_conflict",
                    f"A path component is not a directory: {value}",
                )
            current = resolved

        return current

    def relative_path(self, path: Path) -> str:
        relative = path.relative_to(self._root)
        return relative.as_posix() if relative.parts else "."

    def _ensure_contained(self, path: Path, value: str) -> None:
        if path != self._root and self._root not in path.parents:
            raise WorkspaceError("invalid_path", f"Path escapes workspace: {value}")

    @staticmethod
    def _validate_relative(value: str) -> Path:
        if not isinstance(value, str) or not value:
            raise WorkspaceError("invalid_path", "Path must be a non-empty string")
        if "\x00" in value:
            raise WorkspaceError("invalid_path", "Path contains a NUL byte")

        path = Path(value)
        if path.is_absolute() or path.drive:
            raise WorkspaceError("invalid_path", "Absolute paths are not allowed")
        if any(part == ".." for part in path.parts):
            raise WorkspaceError("invalid_path", "Parent traversal is not allowed")
        return path

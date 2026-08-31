"""Bounded validation for local visual sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .workspace import Workspace

MAX_IMAGE_PATH_CHARS = 1_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_IMAGE_TYPES = {
    ".png": ("png", "image/png"),
    ".jpg": ("jpeg", "image/jpeg"),
    ".jpeg": ("jpeg", "image/jpeg"),
    ".webp": ("webp", "image/webp"),
}


class VisualSourceError(Exception):
    """One safe, stable visual-source validation failure."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class VisualAttachment:
    """Role-neutral public metadata for one validated image."""

    source_label: str
    image_type: str
    mime_type: str
    size_bytes: int

    def to_result_data(self) -> dict[str, str | int]:
        return {
            "path": self.source_label,
            "image_type": self.image_type,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }


class VisualSourceAdapter:
    """Resolve and validate one Workspace image without interpreting it."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def load(self, source_label: str) -> tuple[VisualAttachment, bytes]:
        if len(source_label) > MAX_IMAGE_PATH_CHARS:
            raise VisualSourceError(
                "invalid_arguments",
                f"path cannot exceed {MAX_IMAGE_PATH_CHARS} characters.",
            )

        path = self._workspace.resolve(source_label, expected="file")
        suffix = PurePosixPath(source_label.replace("\\", "/")).suffix.casefold()
        image_format = _IMAGE_TYPES.get(suffix)
        if image_format is None:
            raise VisualSourceError(
                "unsupported_image_type",
                "read_image supports only .png, .jpg, .jpeg, and .webp files.",
            )

        raw_bytes = _bounded_read(path)
        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise VisualSourceError(
                "image_too_large",
                f"Image exceeds the {MAX_IMAGE_BYTES} byte limit.",
            )
        image_type, mime_type = image_format
        if not _matches_signature(raw_bytes, image_type):
            raise VisualSourceError(
                "malformed_image",
                "Image content does not match its supported file type.",
            )

        attachment = VisualAttachment(
            source_label=self._workspace.relative_path(path),
            image_type=image_type,
            mime_type=mime_type,
            size_bytes=len(raw_bytes),
        )
        return attachment, raw_bytes


def _bounded_read(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(MAX_IMAGE_BYTES + 1)


def _matches_signature(raw_bytes: bytes, image_type: str) -> bool:
    if image_type == "png":
        return raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if image_type == "jpeg":
        return raw_bytes.startswith(b"\xff\xd8\xff")
    if image_type == "webp":
        return (
            len(raw_bytes) >= 12
            and raw_bytes[:4] == b"RIFF"
            and raw_bytes[8:12] == b"WEBP"
        )
    return False

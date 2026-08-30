"""Plain per-command dependency confirmation owned by Interaction."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from ..execution.command_policy import CommandApprovalRequest


class ConsoleCommandApprover:
    """Ask once for one dependency-changing command; default to denial."""

    def __init__(
        self,
        *,
        read_line: Callable[[str], str] | None = None,
        write_line: Callable[[str], None] | None = None,
    ) -> None:
        self._read_line = read_line or input
        self._write_line = write_line or print

    def __call__(self, request: CommandApprovalRequest) -> bool:
        self._write_line("⚠ Dependency change")
        self._write_line(f"  {subprocess.list2cmdline(request.command)}")
        self._write_line(f"  {request.reason}")
        response = self._read_line("Proceed? [y/N]: ")
        return response.strip().casefold() in {"y", "yes"}

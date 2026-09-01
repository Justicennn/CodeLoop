"""Local execution layer public surface."""

from .command_policy import (
    CommandDescription,
    CommandTestScope,
    describe_command,
    is_dependency_mutation,
)
from .tools import CommandPreflight, ToolRegistry, ToolResult
from .visual_sources import VisualAttachment, VisualSourceAdapter
from .workspace import Workspace, WorkspaceError

__all__ = [
    "CommandDescription",
    "CommandTestScope",
    "CommandPreflight",
    "ToolRegistry",
    "ToolResult",
    "VisualAttachment",
    "VisualSourceAdapter",
    "Workspace",
    "WorkspaceError",
    "describe_command",
    "is_dependency_mutation",
]

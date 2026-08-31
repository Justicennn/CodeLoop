"""Local execution layer public surface."""

from .command_policy import CommandApprovalRequest, dependency_mutation_request
from .tools import ToolRegistry, ToolResult
from .visual_sources import VisualAttachment, VisualSourceAdapter
from .workspace import Workspace, WorkspaceError

__all__ = [
    "CommandApprovalRequest",
    "ToolRegistry",
    "ToolResult",
    "VisualAttachment",
    "VisualSourceAdapter",
    "Workspace",
    "WorkspaceError",
    "dependency_mutation_request",
]

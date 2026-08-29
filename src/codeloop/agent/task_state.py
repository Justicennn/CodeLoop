"""Explicit state owned by one AgentRunner run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .plan import TaskPlan


@dataclass
class TaskState:
    """Aggregate only the persistent state implemented by the current stage."""

    plan: TaskPlan | None = None

    def replace_plan(self, plan: TaskPlan) -> None:
        self.plan = plan

    def snapshot_for_model(self) -> dict[str, Any] | None:
        if self.plan is None:
            return None
        return {"plan": self.plan.to_snapshot()}

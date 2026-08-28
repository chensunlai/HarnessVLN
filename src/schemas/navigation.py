from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class NavigationTask:
    task_id: str
    instruction: str
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.instruction.strip():
            raise ValueError("instruction must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> JsonObject:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Terminal:
    status: str
    reason: str
    actor: str

    def __post_init__(self) -> None:
        if not self.status.strip():
            raise ValueError("terminal status must not be empty")
        if not self.actor.strip():
            raise ValueError("terminal actor must not be empty")

    def as_dict(self) -> JsonObject:
        return {
            "status": self.status,
            "reason": self.reason,
            "actor": self.actor,
        }

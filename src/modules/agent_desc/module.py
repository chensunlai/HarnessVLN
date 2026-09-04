from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modules.navigation_subagent import NavigationSubagent


class AgentDescModule(NavigationSubagent):
    """Describe the observed local environment without moving."""

    function_name = "agent_desc.run"
    description = (
        "Describe the current local place, boundaries, task-relevant targets, and "
        "visibly supported destinations without moving."
    )
    parameters = {
        "type": "object",
        "properties": {
            "focus": {"type": "string"},
        },
        "additionalProperties": False,
    }
    moves_robot = False

    def execute(self, focus: str = "", **_: Any) -> Mapping[str, Any]:
        return self.unconfigured(
            "agent_desc policy is not configured",
            evidence={"focus": focus},
        )

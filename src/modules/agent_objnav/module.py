from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modules.navigation_subagent import NavigationSubagent


class AgentObjNavModule(NavigationSubagent):
    """Approach one currently visible target without open-ended search."""

    function_name = "agent_objnav.run"
    description = (
        "Approach a target that is already visible and remain visually locked on it. "
        "Refuse if the target is lost."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target": {
                "type": "object",
                "description": "Normalized text, image, or referenced target evidence.",
                "additionalProperties": True,
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    }

    def execute(self, target: Mapping[str, Any], **_: Any) -> Mapping[str, Any]:
        return self.unconfigured(
            "agent_objnav policy is not configured",
            evidence={
                "target_type": target.get("type"),
                "target_fields": sorted(str(key) for key in target),
            },
        )

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modules.navigation_subagent import NavigationSubagent


class AgentLocalModule(NavigationSubagent):
    """Reach one local pose using nearby occupancy information."""

    function_name = "agent_local.run"
    description = (
        "Reach a local target position and orientation using a nearby top-down "
        "occupancy representation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target_pose": {
                "type": "object",
                "properties": {
                    "frame": {"type": "string", "minLength": 1},
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 3,
                    },
                    "heading_degrees": {"type": "number"},
                    "tolerance_m": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["frame", "position", "heading_degrees"],
                "additionalProperties": False,
            },
        },
        "required": ["target_pose"],
        "additionalProperties": False,
    }

    def execute(self, target_pose: Mapping[str, Any], **_: Any) -> Mapping[str, Any]:
        return self.unconfigured(
            "agent_local policy is not configured",
            evidence={"target_pose": dict(target_pose)},
        )

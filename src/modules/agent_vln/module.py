from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modules.navigation_subagent import NavigationSubagent


class AgentVLNModule(NavigationSubagent):
    """Execute one visually verifiable local route instruction."""

    function_name = "agent_vln.run"
    description = (
        "Execute a local natural-language route while each next segment remains "
        "visually verifiable. Refuse when the route no longer matches the scene."
    )
    parameters = {
        "type": "object",
        "properties": {
            "instruction": {"type": "string", "minLength": 1},
        },
        "required": ["instruction"],
        "additionalProperties": False,
    }

    def execute(self, instruction: str, **_: Any) -> Mapping[str, Any]:
        return self.unconfigured(
            "agent_vln policy is not configured",
            evidence={"instruction": instruction},
        )

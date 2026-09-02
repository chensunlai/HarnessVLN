from __future__ import annotations

import time
from typing import Any

from domain.errors import HarnessError
from domain.modules import Module


class ExpertTrajectoryModule(Module):
    """Dummy active module that drives env.step from episode ground truth."""

    def __init__(self, *, delay_s: float = 0.0, stop_status: str = "completed") -> None:
        if delay_s < 0:
            raise ValueError("delay_s must not be negative")
        self.delay_s = delay_s
        self.stop_status = stop_status
        self._trace: list[dict[str, Any]] = []
        self._state = "created"

    def mount(self) -> None:
        self.context.register.register_reference(
            self.context.name,
            f"{self.context.name}.state",
            lambda: self._state,
            description="Expert module lifecycle state.",
        )

    def run(self) -> None:
        actions = self.context.episode.truth.get("expert_actions")
        if not isinstance(actions, (list, tuple)):
            raise HarnessError("ExpertTrajectoryModule requires truth.expert_actions")
        self._state = "running"
        for index, action in enumerate(actions):
            if self.context.cancelled.is_set():
                return
            result = self.context.register.call(
                self.context.name, "env.step", {"action": str(action)}
            )
            self._trace.append({"index": index, "action": action, "result": result})
            if self.delay_s:
                time.sleep(self.delay_s)
        self.context.output.write_json("trajectory.json", self._trace)
        self.context.register.call(
            self.context.name,
            "env.stop",
            {
                "status": self.stop_status,
                "reason": "expert trajectory completed",
                "actor": self.context.name,
            },
        )
        self._state = "finished"

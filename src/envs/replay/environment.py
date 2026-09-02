from __future__ import annotations

import threading
import time
from typing import Any, Mapping

from domain.errors import HarnessError
from domain.modules import EnvironmentModule


class ReplayEnvironment(EnvironmentModule):
    """Deterministic navigation environment used to validate a Domain composition."""

    def __init__(self, *, max_steps: int = 256, step_delay_s: float = 0.0) -> None:
        super().__init__()
        if max_steps < 1 or step_delay_s < 0:
            raise ValueError("invalid Replay environment limits")
        self.max_steps = max_steps
        self.step_delay_s = step_delay_s
        self._lock = threading.RLock()
        self._actions: list[str] = []
        self._expert: tuple[str, ...] = ()
        self._observation_id = 0
        self._pose = [0.0, 0.0, 0.0]
        self._stopped = False

    def mount(self) -> None:
        register = self.context.register
        self.expose(
            "env.observe",
            self.observe,
            description="Read the current normalized navigation observation.",
            parameters={"type": "object", "additionalProperties": False},
        )
        self.expose(
            "env.step",
            self.step,
            description="Execute one discrete navigation action.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "enum": ["forward", "backward", "left", "right", "look_up", "look_down"]
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            mutates=True,
            serial_key="env.state",
        )
        register.register_reference(
            self.context.name,
            "env.observation",
            self.observe,
            description="Latest Replay observation.",
        )
        register.register_reference(
            self.context.name,
            "env.instruction",
            lambda: dict(self.context.episode.instruction),
            description="Current normalized navigation instruction.",
        )
        register.register_reference(
            self.context.name,
            "env.pose",
            lambda: list(self._pose),
            description="Current Replay pose.",
        )

    def start(self) -> None:
        self._expert = tuple(self.context.episode.truth.get("expert_actions", ()))
        start_pose = self.context.episode.setup.get("start_pose", (0, 0, 0))
        if len(start_pose) != 3:
            raise HarnessError("Replay start_pose must contain x, y, yaw")
        self._pose = [float(value) for value in start_pose]
        self.context.metadata.update(
            {"backend": "replay", "max_steps": self.max_steps}
        )

    def observe(self) -> dict[str, Any]:
        with self._lock:
            self._observation_id += 1
            return {
                "observation_id": self._observation_id,
                "instruction": dict(self.context.episode.instruction),
                "pose": list(self._pose),
                "step_count": len(self._actions),
            }

    def step(self, action: str) -> dict[str, Any]:
        if not self.wait_ready(0):
            raise HarnessError("Replay environment is not ready")
        if self.wait_terminal(0) is not None:
            raise HarnessError("Replay environment is stopped")
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        with self._lock:
            if len(self._actions) >= self.max_steps:
                self.finish("failed", "maximum step count reached", "env")
                return {"accepted": False, "terminal": True}
            self._actions.append(action)
            self._apply(action)
            return {
                "accepted": True,
                "action": action,
                "step_count": len(self._actions),
                "pose": list(self._pose),
            }

    def stop(self, reason: str) -> None:
        del reason
        self._stopped = True

    def result(self) -> Mapping[str, Any]:
        with self._lock:
            matched = sum(
                actual == expected
                for actual, expected in zip(self._actions, self._expert)
            )
            success = bool(self._expert) and tuple(self._actions) == self._expert
            if not self._expert:
                success = True
            return {
                "success": success,
                "action_count": len(self._actions),
                "expert_action_count": len(self._expert),
                "matched_actions": matched,
                "actions": list(self._actions),
                "final_pose": list(self._pose),
                "stopped": self._stopped,
            }

    def _apply(self, action: str) -> None:
        if action == "forward":
            self._pose[1] += 1.0
        elif action == "backward":
            self._pose[1] -= 1.0
        elif action == "left":
            self._pose[2] = (self._pose[2] - 90.0) % 360.0
        elif action == "right":
            self._pose[2] = (self._pose[2] + 90.0) % 360.0

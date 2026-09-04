from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from domain.errors import HarnessError
from domain.io import json_value
from domain.modules import Module


SUBTASK_STATUSES = frozenset({"completed", "refused", "failed"})


class NavigationSubagent(Module, ABC):
    """Common function boundary for one train-free navigation capability."""

    function_name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[Mapping[str, Any]]
    moves_robot: ClassVar[bool] = True

    def __init__(self) -> None:
        self._call_count = 0

    def mount(self) -> None:
        self.expose(
            self.function_name,
            self._invoke,
            description=self.description,
            parameters=self.parameters,
            mutates=self.moves_robot,
            serial_key="navigation.motion" if self.moves_robot else None,
        )
        self.context.metadata.update(
            {
                "role": "navigation_subagent",
                "function": self.function_name,
                "moves_robot": self.moves_robot,
                "policy": "stub",
            }
        )

    def _invoke(self, **arguments: Any) -> dict[str, Any]:
        result = self.execute(**arguments)
        if not isinstance(result, Mapping):
            raise HarnessError(f"{self.function_name} must return an object")
        value = dict(result)
        status = value.get("status")
        if status not in SUBTASK_STATUSES:
            raise HarnessError(
                f"{self.function_name} returned invalid subtask status {status!r}"
            )
        if not isinstance(value.get("reason"), str):
            raise HarnessError(f"{self.function_name} must return a reason string")
        if not isinstance(value.get("evidence"), Mapping):
            raise HarnessError(f"{self.function_name} must return evidence as an object")
        self._call_count += 1
        self.context.metadata["calls"] = self._call_count
        self.context.output.write_json(
            "last_call.json",
            json_value({"arguments": arguments, "result": value}),
        )
        return value

    @abstractmethod
    def execute(self, **arguments: Any) -> Mapping[str, Any]:
        raise NotImplementedError

    def observe(self) -> Mapping[str, Any]:
        value = self.context.register.call(self.context.name, "env.observe")
        return value if isinstance(value, Mapping) else {}

    def unconfigured(
        self,
        reason: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        observation = self.observe()
        value: dict[str, Any] = {
            "status": "refused",
            "reason": reason,
            "evidence": {
                **dict(evidence or {}),
                "observation": observation_evidence(observation),
            },
        }
        pose = observation.get("pose")
        if pose is not None:
            value["final_pose"] = json_value(pose)
        return value


def observation_evidence(observation: Mapping[str, Any]) -> dict[str, Any]:
    channels = observation.get("channels")
    return {
        "observation_id": observation.get("observation_id"),
        "step": observation.get("step", observation.get("step_count")),
        "pose": json_value(observation.get("pose")),
        "channels": sorted(channels) if isinstance(channels, Mapping) else [],
    }

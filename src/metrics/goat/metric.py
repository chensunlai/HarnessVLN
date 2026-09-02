from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from domain.errors import HarnessError
from domain.modules import MetricModule


class GOATMetric(MetricModule):
    def __init__(self) -> None:
        self._values: dict[str, float] = {}

    def mount(self) -> None:
        self.context.register.register_reference(
            self.context.name,
            "metric.values",
            lambda: dict(self._values),
            description="GOAT per-episode metrics.",
        )

    def evaluate(self, terminal, environment: Mapping[str, Any]) -> Mapping[str, float]:
        del terminal
        goals = environment.get("goal_results")
        if not isinstance(goals, Sequence) or isinstance(goals, (str, bytes)) or not goals:
            raise HarnessError("GOAT environment result requires goal_results")
        values: dict[str, float] = {
            "success": _mean(goals, "success"),
            "spl": _mean(goals, "spl"),
        }
        for modality in ("object", "description", "image"):
            selected = [
                goal
                for goal in goals
                if isinstance(goal, Mapping) and goal.get("modality") == modality
            ]
            if selected:
                values[f"success_{modality}"] = _mean(selected, "success")
                values[f"spl_{modality}"] = _mean(selected, "spl")
        self._values = values
        self.context.output.write_json("metrics.json", values)
        return values


def _mean(values: Sequence[Any], key: str) -> float:
    try:
        return sum(float(item[key]) for item in values) / len(values)
    except (KeyError, TypeError, ValueError) as error:
        raise HarnessError(f"GOAT goal results have no numeric {key!r}") from error

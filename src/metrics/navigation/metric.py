from __future__ import annotations

from typing import Any, Mapping

from domain.modules import MetricModule


class NavigationMetric(MetricModule):
    """Small metric adapter for Replay and native navigation result fields."""

    def __init__(self, *, fields: Mapping[str, str] | None = None) -> None:
        self.fields = dict(fields or {})
        self._values: dict[str, float] = {}

    def mount(self) -> None:
        self.context.register.register_reference(
            self.context.name,
            "metric.values",
            lambda: dict(self._values),
            description="Metrics computed for the current episode.",
        )

    def evaluate(self, terminal, environment: Mapping[str, Any]) -> Mapping[str, float]:
        del terminal
        if self.fields:
            values = {}
            for output, source in self.fields.items():
                value = _lookup(environment, source)
                if value is not None:
                    values[output] = float(value)
        else:
            success = float(bool(environment.get("success", False)))
            actual = float(environment.get("action_count", 0))
            expert = float(environment.get("expert_action_count", actual))
            denominator = max(actual, expert)
            values = {
                "success": success,
                "path_efficiency": success * (expert / denominator if denominator else 1.0),
            }
            for name in ("spl", "distance_to_goal", "ndtw", "oracle_success"):
                if name in environment:
                    values[name] = float(environment[name])
        self._values = values
        self.context.output.write_json("metrics.json", values)
        return values


def _lookup(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current

from __future__ import annotations

from harness import ComponentContext, Metric, Terminal


class DummyMetric(Metric):
    name = "metric"
    required_functions = frozenset({"env.metric_state"})

    async def start(self, context: ComponentContext) -> None:
        self._context = context

    async def evaluate(self, terminal: Terminal, environment):
        state = await self._context.functions.call("env.metric_state")
        distance = abs(state["target"] - state["position"])
        shortest = abs(state["target"] - state["start"])
        travelled = max(0, len(state["trajectory"]) - 1)
        success = float(terminal.status == "completed" and distance == 0)
        spl = success if shortest == 0 else success * shortest / max(shortest, travelled)
        values = {"success": success, "spl": spl, "distance": float(distance)}
        self._context.output.write_json("metrics.json", values)
        self._context.output.add_artifact("metrics.json", "application/json")
        return values

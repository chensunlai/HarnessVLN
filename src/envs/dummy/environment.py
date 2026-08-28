from __future__ import annotations

import asyncio

from harness import ComponentContext, Environment, Function, Terminal
from schemas import NAV_STOP, nav_stop_input_schema, nav_stop_output_schema


_EMPTY = {"type": "object", "additionalProperties": False}
_MOVE = {
    "type": "object",
    "properties": {"delta": {"type": "integer", "minimum": -1, "maximum": 1}},
    "required": ["delta"],
    "additionalProperties": False,
}
class DummyEnvironment(Environment):
    """A deterministic one-dimensional environment used to validate the framework."""

    name = "environment"

    def __init__(self, start: int = 0, max_steps: int = 100) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.start_position = start
        self.max_steps = max_steps
        self.position = start
        self.target = start
        self.trajectory = [start]
        self._terminal: Terminal | None = None
        self._terminal_event = asyncio.Event()
        self._context: ComponentContext | None = None

    def functions(self):
        return (
            Function(
                "nav.observe",
                "Return the latest public navigation observation.",
                self._observe,
                input_schema=_EMPTY,
            ),
            Function(
                "nav.move",
                "Move one unit left, right, or remain still.",
                self._move,
                input_schema=_MOVE,
                mutates=True,
                serial_key="environment",
            ),
            Function(
                NAV_STOP,
                "End this environment session.",
                self._stop,
                input_schema=nav_stop_input_schema(),
                output_schema=nav_stop_output_schema(),
                mutates=True,
                serial_key="environment",
            ),
            Function(
                "env.metric_state",
                "Return evaluator-only trajectory and goal state.",
                self._metric_state,
                input_schema=_EMPTY,
            ),
        )

    async def start(self, context: ComponentContext) -> None:
        self._context = context
        self.target = int(context.task.metadata.get("target", self.start_position))
        context.output.set_metadata(
            backend="dummy", start=self.start_position, target=self.target
        )

    async def _observe(self, _call, _arguments):
        return {"position": self.position, "step": len(self.trajectory) - 1}

    async def _move(self, _call, arguments):
        if self._terminal is not None:
            raise RuntimeError("the environment is already terminal")
        self.position += arguments["delta"]
        self.trajectory.append(self.position)
        if len(self.trajectory) - 1 >= self.max_steps and self.position != self.target:
            self._claim_terminal("failed", "environment step limit reached", self.name)
        return {"position": self.position, "step": len(self.trajectory) - 1}

    async def _stop(self, _call, arguments):
        terminal = self._claim_terminal(
            arguments["status"], arguments["reason"], arguments["actor"]
        )
        return terminal.as_dict()

    async def _metric_state(self, _call, _arguments):
        return {
            "start": self.start_position,
            "target": self.target,
            "position": self.position,
            "trajectory": list(self.trajectory),
        }

    def _claim_terminal(self, status: str, reason: str, actor: str) -> Terminal:
        if self._terminal is None:
            self._terminal = Terminal(status, reason, actor)
            self._terminal_event.set()
        return self._terminal

    async def wait_terminal(self) -> Terminal:
        await self._terminal_event.wait()
        assert self._terminal is not None
        return self._terminal

    async def result(self):
        return {
            "position": self.position,
            "steps": len(self.trajectory) - 1,
            "trajectory": list(self.trajectory),
        }

    async def close(self, reason: str) -> None:
        if self._context is None:
            return
        self._context.output.write_json(
            "trajectory.json",
            {
                "trajectory": self.trajectory,
                "target": self.target,
                "terminal": self._terminal.as_dict() if self._terminal else None,
            },
        )
        self._context.output.add_artifact("trajectory.json", "application/json")

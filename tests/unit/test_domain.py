import asyncio
import json

import pytest

from harness import (
    Agent,
    Component,
    DomainComponents,
    DomainRuntime,
    Environment,
    Function,
    Metric,
    NavigationTask,
    Terminal,
)
from harness.errors import ContractError


EMPTY_INPUT = {"type": "object", "additionalProperties": False}
STOP_INPUT = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "reason": {"type": "string"},
        "actor": {"type": "string"},
    },
    "required": ["status", "reason", "actor"],
    "additionalProperties": False,
}


def run(operation):
    return asyncio.run(operation)


class TestEnvironment(Environment):
    __test__ = False
    name = "environment"

    def __init__(self, closed):
        self.position = 0
        self.moves = 0
        self._terminal = None
        self._terminal_event = asyncio.Event()
        self.closed = closed

    def functions(self):
        return (
            Function(
                "nav.observe",
                "Read the current position.",
                self._observe,
                input_schema=EMPTY_INPUT,
            ),
            Function(
                "nav.move",
                "Move by a signed distance.",
                self._move,
                input_schema={
                    "type": "object",
                    "properties": {"distance": {"type": "integer"}},
                    "required": ["distance"],
                    "additionalProperties": False,
                },
                mutates=True,
                serial_key="environment",
            ),
            Function(
                "nav.stop",
                "Stop this navigation Domain.",
                self._stop,
                input_schema=STOP_INPUT,
                mutates=True,
                serial_key="environment",
            ),
            Function(
                "env.metric_state",
                "Read evaluator-only environment state.",
                self._metric_state,
                input_schema=EMPTY_INPUT,
            ),
        )

    async def _observe(self, _call, _arguments):
        return {"position": self.position}

    async def _move(self, _call, arguments):
        if self._terminal is not None:
            raise RuntimeError("environment already stopped")
        self.position += arguments["distance"]
        self.moves += 1
        return {"position": self.position}

    async def _stop(self, call, arguments):
        if self._terminal is None:
            self._terminal = Terminal(
                arguments["status"], arguments["reason"], arguments["actor"]
            )
            self._terminal_event.set()
        return self._terminal.as_dict()

    async def _metric_state(self, _call, _arguments):
        return {"target": 2, "position": self.position}

    async def wait_terminal(self):
        await self._terminal_event.wait()
        return self._terminal

    async def result(self):
        return {"position": self.position, "moves": self.moves}

    async def close(self, reason):
        self.closed.append((self.name, reason))


class ScriptAgent(Agent):
    name = "agent"
    required_functions = frozenset({"nav.observe", "nav.move", "nav.stop"})

    def __init__(self, closed):
        self.closed = closed
        self.runs = 0

    async def run(self, context):
        self.runs += 1
        assert "env.metric_state" not in {spec.name for spec in context.functions.specs}
        assert await context.functions.call("nav.observe") == {"position": 0}
        await context.functions.call("nav.move", distance=2)
        context.output.append_jsonl("model/trace.jsonl", {"text": "done"})
        context.output.add_artifact("model/trace.jsonl", "application/jsonl")
        await context.functions.call(
            "nav.stop", status="completed", reason="goal reached", actor="agent"
        )

    async def close(self, reason):
        self.closed.append((self.name, reason))


class DistanceMetric(Metric):
    name = "distance_metric"
    required_functions = frozenset({"env.metric_state"})

    async def start(self, context):
        self.context = context

    async def evaluate(self, terminal, environment):
        state = await self.context.functions.call("env.metric_state")
        assert environment["position"] == state["position"]
        return {
            "success": float(terminal.status == "completed"),
            "distance": abs(state["target"] - state["position"]),
        }


class PassiveService(Component):
    name = "service"

    def __init__(self, closed):
        self.closed = closed

    async def close(self, reason):
        self.closed.append((self.name, reason))


def test_agent_drives_one_complete_domain_and_components_own_outputs(tmp_path):
    closed = []
    environment = TestEnvironment(closed)
    agent = ScriptAgent(closed)
    metric = DistanceMetric()
    service = PassiveService(closed)
    task = NavigationTask("case-1", "Move to position two")

    result = run(
        DomainRuntime(timeout_s=1).run(
            task,
            DomainComponents(environment, agent, (service,), (metric,)),
            output_root=str(tmp_path),
            domain_id="domain-1",
        )
    )

    assert agent.runs == 1
    assert result.terminal == Terminal("completed", "goal reached", "agent")
    assert result.environment == {"position": 2, "moves": 1}
    assert result.metrics == {"success": 1.0, "distance": 0.0}
    assert result.cleanup_errors == ()
    assert [name for name, _ in closed] == ["agent", "service", "environment"]

    root = tmp_path / "domain-1"
    assert json.loads((root / "domain.json").read_text())["metrics"]["success"] == 1.0
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    assert [call["function"] for call in calls] == [
        "nav.observe",
        "nav.move",
        "nav.stop",
        "env.metric_state",
    ]
    trace = root / "components" / "agent" / "model" / "trace.jsonl"
    assert json.loads(trace.read_text()) == {"text": "done"}
    assert result.components["agent"].artifacts[0].path == "model/trace.jsonl"


class ReturningAgent(Agent):
    name = "agent"

    async def run(self, context):
        return None


def test_agent_return_without_stop_becomes_protocol_failure():
    environment = TestEnvironment([])
    result = run(
        DomainRuntime(timeout_s=1).run(
            NavigationTask("case-2", "Do not stop"),
            DomainComponents(environment, ReturningAgent()),
        )
    )
    assert result.terminal.status == "failed"
    assert result.terminal.actor == "domain"
    assert "without stopping" in result.terminal.reason


class MissingDependencyAgent(ReturningAgent):
    required_functions = frozenset({"vln.missing"})


def test_missing_component_dependency_fails_before_environment_start():
    with pytest.raises(ContractError, match="vln.missing"):
        run(
            DomainRuntime().run(
                NavigationTask("case-3", "Use missing function"),
                DomainComponents(TestEnvironment([]), MissingDependencyAgent()),
            )
        )


class NoStopEnvironment(TestEnvironment):
    def functions(self):
        return tuple(
            function for function in super().functions() if function.name != "nav.stop"
        )


class FakeStopService(Component):
    name = "fake_stop"

    def functions(self):
        async def stop(_call, arguments):
            return arguments

        return (Function("nav.stop", "Fake stop.", stop, input_schema=STOP_INPUT),)


def test_nav_stop_must_be_owned_by_environment():
    with pytest.raises(ContractError, match="provided by the environment"):
        run(
            DomainRuntime().run(
                NavigationTask("case-4", "Reject fake stop"),
                DomainComponents(
                    NoStopEnvironment([]), ReturningAgent(), (FakeStopService(),)
                ),
            )
        )

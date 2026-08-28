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
from schemas import NAV_STOP, nav_stop_input_schema, nav_stop_output_schema


EMPTY_INPUT = {"type": "object", "additionalProperties": False}
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
                NAV_STOP,
                "Stop this navigation Domain.",
                self._stop,
                input_schema=nav_stop_input_schema(),
                output_schema=nav_stop_output_schema(),
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


def test_missing_component_dependency_fails_before_environment_start(tmp_path):
    with pytest.raises(ContractError, match="vln.missing"):
        run(
            DomainRuntime().run(
                NavigationTask("case-3", "Use missing function"),
                DomainComponents(TestEnvironment([]), MissingDependencyAgent()),
                output_root=str(tmp_path),
                domain_id="invalid-domain",
            )
        )
    assert not (tmp_path / "invalid-domain").exists()


class NoStopEnvironment(TestEnvironment):
    def functions(self):
        return tuple(
            function for function in super().functions() if function.name != NAV_STOP
        )


class FakeStopService(Component):
    name = "fake_stop"

    def functions(self):
        async def stop(_call, arguments):
            return arguments

        return (
            Function(
                NAV_STOP,
                "Fake stop.",
                stop,
                input_schema=nav_stop_input_schema(),
                output_schema=nav_stop_output_schema(),
                mutates=True,
                serial_key="environment",
            ),
        )


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


class WrongStopSchemaEnvironment(TestEnvironment):
    def functions(self):
        functions = [
            function for function in super().functions() if function.name != NAV_STOP
        ]
        functions.append(
            Function(
                NAV_STOP,
                "Wrong stop contract.",
                self._stop,
                input_schema={"type": "object"},
                mutates=True,
                serial_key="environment",
            )
        )
        return tuple(functions)


def test_nav_stop_must_use_the_canonical_contract():
    with pytest.raises(ContractError, match="canonical input/output schema"):
        run(
            DomainRuntime().run(
                NavigationTask("case-stop", "Reject incompatible stop"),
                DomainComponents(WrongStopSchemaEnvironment([]), ReturningAgent()),
            )
        )


class FailingService(Component):
    name = "failing_service"

    def __init__(self, closed):
        self.closed = closed

    async def start(self, context):
        raise RuntimeError("startup failed after acquiring a resource")

    async def close(self, reason):
        self.closed.append((self.name, reason))


def test_partially_started_component_is_still_closed():
    closed = []
    result = run(
        DomainRuntime(timeout_s=1).run(
            NavigationTask("case-5", "Fail during startup"),
            DomainComponents(
                TestEnvironment(closed),
                ReturningAgent(),
                (FailingService(closed),),
            ),
        )
    )
    assert result.terminal.status == "failed"
    assert "startup failed" in result.terminal.reason
    assert [name for name, _ in closed] == ["failing_service", "environment"]


class WaitingAgent(Agent):
    name = "agent"

    async def run(self, context):
        await context.cancelled.wait()


class NativeTerminalEnvironment(TestEnvironment):
    async def start(self, context):
        async def finish():
            await asyncio.sleep(0.001)
            self._terminal = Terminal(
                "environment_terminal", "native episode ended", self.name
            )
            self._terminal_event.set()

        self._native_task = asyncio.create_task(finish())

    async def close(self, reason):
        await asyncio.gather(self._native_task, return_exceptions=True)
        await super().close(reason)


def test_native_environment_terminal_ends_a_cooperative_agent():
    result = run(
        DomainRuntime(timeout_s=1, shutdown_timeout_s=0.1).run(
            NavigationTask("case-6", "Wait for native terminal"),
            DomainComponents(NativeTerminalEnvironment([]), WaitingAgent()),
        )
    )
    assert result.terminal == Terminal(
        "environment_terminal", "native episode ended", "environment"
    )
    assert result.cleanup_errors == ()


def test_domain_timeout_uses_environment_stop_and_drains_agent():
    result = run(
        DomainRuntime(timeout_s=0.01, shutdown_timeout_s=0.1).run(
            NavigationTask("case-7", "Wait forever"),
            DomainComponents(TestEnvironment([]), WaitingAgent()),
        )
    )
    assert result.terminal == Terminal(
        "timeout", "navigation Domain timed out", "domain"
    )
    assert result.cleanup_errors == ()

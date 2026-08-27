from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from agents import NormalAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.output import RunOutput
from schemas import NavGoal, NavTask


class ScriptedResponses:
    def __init__(self, outputs: list[list[Any]]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        captured = dict(request)
        captured["input"] = list(request["input"])
        self.requests.append(captured)
        output = self.outputs.pop(0)
        response_id = f"response-{len(self.requests)}"
        response_record = {
            "id": response_id,
            "model": request["model"],
            "status": "completed",
            "output": [vars(item) for item in output],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        }
        return SimpleNamespace(
            output=output,
            model_dump=lambda **_: response_record,
        )


class TransientModelError(Exception):
    status_code = 502


def function_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_normal_agent_runs_native_responses_tool_loop() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("call-1", "nav__observe", {})],
                [
                    function_call(
                        "call-2", "nav__move__discrete", {"action": "forward"}
                    ),
                    function_call(
                        "call-3", "nav__move__discrete", {"action": "forward"}
                    ),
                ],
                [
                    function_call(
                        "call-4", "nav__goal__finish", {"status": "success"}
                    )
                ],
                [
                    function_call(
                        "call-5",
                        "nav__stop",
                        {"status": "success", "reason": "done"},
                    )
                ],
            ]
        )
        client = SimpleNamespace(responses=responses)
        agent = NormalAgent(
            "test-model",
            ("nav.observe", "nav.move.discrete", "nav.goal.finish"),
            guidance="Fixture-specific guidance.",
            reasoning_effort="medium",
            client=client,
        )
        goal = NavGoal("goal", "move to the marker")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("normal", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(2,)),
            ),
        )

        assert result.terminal.status == "completed"
        assert result.environment["position"] == 2
        assert [event.name for event in result.audit] == [
            "nav.observe",
            "nav.move.discrete",
            "nav.move.discrete",
            "nav.goal.finish",
            "nav.stop",
        ]
        assert all(event.actor == "agent" for event in result.audit)
        assert result.audit[-2].arguments["status"] == "completed"
        assert result.audit[-1].arguments["status"] == "completed"

        first_request = responses.requests[0]
        assert first_request["parallel_tool_calls"] is True
        assert first_request["reasoning"] == {"effort": "medium"}
        assert first_request["instructions"].endswith(
            "Fixture-specific guidance."
        )
        assert first_request["store"] is False
        assert first_request["tool_choice"] == "required"
        assert {tool["name"] for tool in first_request["tools"]} == {
            "nav__goal__finish",
            "nav__move__discrete",
            "nav__observe",
            "nav__stop",
        }
        assert all("function" not in tool for tool in first_request["tools"])
        assert all(tool["strict"] is False for tool in first_request["tools"])

        observation_output = responses.requests[1]["input"][-1]
        assert observation_output["type"] == "function_call_output"
        assert observation_output["call_id"] == "call-1"
        assert json.loads(observation_output["output"])["ok"] is True

    run(scenario())


def test_normal_agent_rejects_mixed_tool_batches_and_recovers() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [
                    function_call("observe", "nav__observe", {}),
                    function_call(
                        "move", "nav__move__discrete", {"action": "forward"}
                    ),
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "batch corrected"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe", "nav.move.discrete"),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "test batching")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("batch", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
            ),
        )

        assert result.terminal.status == "completed"
        assert [event.name for event in result.audit] == ["nav.stop"]
        rejected = responses.requests[1]["input"][-2:]
        assert {item["call_id"] for item in rejected} == {"observe", "move"}
        assert all(
            json.loads(item["output"])["error"]["type"] == "ToolBatchError"
            for item in rejected
        )

    run(scenario())


def test_normal_agent_enforces_four_action_hard_limit() -> None:
    try:
        NormalAgent("test-model", ("nav.observe",), max_actions_per_turn=5)
    except ValueError as error:
        assert "between 1 and 4" in str(error)
    else:
        raise AssertionError("NormalAgent accepted more than four actions per turn")


def test_normal_agent_rejects_five_model_actions_without_executing_them() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [
                    function_call(
                        f"move-{index}",
                        "nav__move__discrete",
                        {"action": "forward"},
                    )
                    for index in range(5)
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "limit handled"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.move.discrete",),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "test the action limit")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("action-limit", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
            ),
        )

        assert result.environment["position"] == 0
        assert [event.name for event in result.audit] == ["nav.stop"]
        rejected = responses.requests[1]["input"][-5:]
        assert all(
            "maximum is 4" in json.loads(item["output"])["error"]["message"]
            for item in rejected
        )

    run(scenario())


def test_normal_agent_returns_rgb_as_native_responses_image() -> None:
    class VisualEnvironment(DummyNavigationEnvironment):
        async def _observe(self, actor, arguments):
            observation = await super()._observe(actor, arguments)
            observation["channels"]["rgb"] = np.zeros(
                (480, 640, 3), dtype=np.uint8
            )
            observation["channels"]["depth"] = np.full(
                (480, 640, 1), 0.25, dtype=np.float32
            )
            return observation

    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "image received"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe",),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "inspect the image")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("visual", goal),
            NavigationStack(
                agent,
                VisualEnvironment((goal,), targets=(0,)),
            ),
        )

        assert result.terminal.status == "completed"
        content = responses.requests[1]["input"][-1]["output"]
        assert content[0]["type"] == "input_text"
        model_output = json.loads(content[0]["text"])
        assert model_output["ok"] is True
        assert model_output["sensor_summary"]["depth"] == {
            "center": 0.25,
            "grid": [[0.25, 0.25, 0.25]] * 3,
            "minimum": 0.25,
            "maximum": 0.25,
            "lower_is_nearer": True,
        }
        assert content[1]["type"] == "input_image"
        assert content[1]["detail"] == "high"
        assert content[1]["image_url"].startswith("data:image/jpeg;base64,")

    run(scenario())


def test_normal_agent_returns_tool_errors_to_the_model() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [
                    function_call(
                        "bad-action", "nav__move__discrete", {"action": "fly"}
                    )
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "error handled"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.move.discrete",),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "test recovery")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("recover", goal),
            NavigationStack(agent, DummyNavigationEnvironment((goal,), targets=(0,))),
        )

        assert result.terminal.status == "completed"
        error_output = json.loads(responses.requests[1]["input"][-1]["output"])
        assert error_output["ok"] is False
        assert error_output["error"]["type"] == "ToolValidationError"
        assert [event.outcome for event in result.audit] == ["invalid", "ok"]

    run(scenario())


def test_normal_agent_retries_transient_model_errors() -> None:
    class RetryResponses:
        def __init__(self) -> None:
            self.calls = 0

        async def create(self, **request: Any) -> Any:
            del request
            self.calls += 1
            if self.calls == 1:
                raise TransientModelError("overloaded")
            return SimpleNamespace(
                output=[
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "retry worked"},
                    )
                ]
            )

    async def scenario() -> None:
        responses = RetryResponses()
        agent = NormalAgent(
            "test-model",
            ("nav.observe",),
            retry_backoff_s=0,
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "test retry")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("retry", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
            ),
        )

        assert responses.calls == 2
        assert result.terminal.status == "completed"

    run(scenario())


def test_normal_agent_stops_when_iteration_budget_is_exhausted() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [[function_call("observe", "nav__observe", {})]]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe",),
            max_iterations=1,
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "test budget")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("budget", goal),
            NavigationStack(agent, DummyNavigationEnvironment((goal,), targets=(0,))),
        )

        assert result.terminal.status == "failed"
        assert result.terminal.reason == "agent iteration budget reached"
        assert [event.name for event in result.audit] == ["nav.observe", "nav.stop"]

    run(scenario())


def test_normal_agent_persists_complete_model_trace_without_inline_images(
    tmp_path,
) -> None:
    class VisualEnvironment(DummyNavigationEnvironment):
        async def _observe(self, actor, arguments):
            observation = await super()._observe(actor, arguments)
            observation["channels"]["rgb"] = np.zeros(
                (12, 16, 3), dtype=np.uint8
            )
            return observation

    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "trace saved"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe",),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "inspect the image")
        task = NavTask("trace", goal)
        run_output = RunOutput(
            {"root": str(tmp_path), "run_id": "trace-run"},
            resolved_config={},
            config_sources=(),
            config_digest="a" * 64,
            provenance={},
        )
        episode = run_output.benchmark(0, "fixture", "test").episode(
            0, "trace", {"case_id": "trace", "task": task}
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            task,
            NavigationStack(agent, VisualEnvironment((goal,), targets=(0,))),
            output=episode,
        )
        episode.finish({"terminal": result.terminal})

        component_dir = episode.path / "components"
        record = json.loads((component_dir / "agent.json").read_text())
        trace_text = (component_dir / "agent.events.jsonl").read_text()
        events = [json.loads(line) for line in trace_text.splitlines()]

        assert record["model_trace"] == {
            "schema_version": 1,
            "format": "jsonl",
            "path": "components/agent.events.jsonl",
        }
        assert record["model_responses"] == 2
        assert record["usage"] == {
            "input_tokens": 20,
            "output_tokens": 4,
            "total_tokens": 24,
        }
        assert [event["sequence"] for event in events] == list(
            range(1, len(events) + 1)
        )
        assert [event["type"] for event in events] == [
            "agent.started",
            "model.request",
            "model.response",
            "tool.result",
            "model.request",
            "model.response",
            "tool.result",
            "agent.terminal",
        ]
        responses_in_trace = [
            event["response"]
            for event in events
            if event["type"] == "model.response"
        ]
        assert responses_in_trace[0]["id"] == "response-1"
        assert responses_in_trace[0]["output"][0]["name"] == "nav__observe"
        observe_result = next(
            event
            for event in events
            if event["type"] == "tool.result"
            and event["tool_name"] == "nav.observe"
        )
        logged_image = observe_result["model_input"]["output"][1]["image_url"]
        assert logged_image["source"] == "nav.observe.channels.rgb"
        assert logged_image["media_type"] == "image/jpeg"
        assert logged_image["encoded_bytes"] > 0
        assert "data:image" not in trace_text

    run(scenario())

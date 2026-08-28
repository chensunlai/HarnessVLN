from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from agents import NormalAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.output import RunOutput
from schemas import NavGoal, NavTask
from vln import DummyVLNNavigator


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


def reasoning_summary(text: str) -> Any:
    return SimpleNamespace(
        type="reasoning",
        summary=[SimpleNamespace(type="summary_text", text=text)],
    )


def commentary(text: str) -> Any:
    return SimpleNamespace(
        type="message",
        phase="commentary",
        content=[SimpleNamespace(type="output_text", text=text)],
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
                [function_call("call-4", "nav__observe", {})],
                [
                    function_call(
                        "call-5", "nav__goal__finish", {"status": "success"}
                    )
                ],
                [
                    function_call(
                        "call-6",
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
            "nav.observe",
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


def test_normal_agent_optionally_requests_and_records_model_explanations(
    tmp_path,
) -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [
                    reasoning_summary("The destination is already reached."),
                    commentary("I can finish from the current position."),
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "already there"},
                    ),
                ]
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe",),
            reasoning_effort="medium",
            reasoning_summary=True,
            commentary=True,
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "remain at the marker")
        task = NavTask("explanations", goal)
        run_output = RunOutput(
            {"root": str(tmp_path), "run_id": "explanations-run"},
            resolved_config={},
            config_sources=(),
            config_digest="a" * 64,
            provenance={},
        )
        episode = run_output.benchmark(0, "fixture", "test").episode(
            0, "explanations", {"case_id": "explanations", "task": task}
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            task,
            NavigationStack(agent, DummyNavigationEnvironment((goal,), targets=(0,))),
            output=episode,
        )
        episode.finish({"terminal": result.terminal})

        request = responses.requests[0]
        assert request["reasoning"] == {"effort": "medium", "summary": "auto"}
        assert "user-visible commentary message" in request["instructions"]

        events = [
            json.loads(line)
            for line in (
                episode.path / "components" / "agent.events.jsonl"
            ).read_text().splitlines()
        ]
        summary_event = next(
            event for event in events if event["type"] == "model.reasoning_summary"
        )
        commentary_event = next(
            event for event in events if event["type"] == "model.commentary"
        )
        assert summary_event["text"] == "The destination is already reached."
        assert commentary_event["iteration"] == 1
        assert commentary_event["phase"] == "commentary"
        assert commentary_event["text"] == "I can finish from the current position."
        assert result.terminal.status == "completed"

    run(scenario())


def test_normal_agent_uses_one_blocking_local_vln_call() -> None:
    async def scenario() -> None:
        local_instruction = (
            "Move to the visible doorway straight ahead and stop just beyond it."
        )
        responses = ScriptedResponses(
            [
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        "local-vln",
                        "vln__navigate__local",
                        {"instruction": local_instruction, "max_steps": 8},
                    )
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "local call returned"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe", "vln.navigate.local"),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "walk through several rooms to the destination")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("local-vln", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
                vln=DummyVLNNavigator(),
            ),
        )

        assert result.terminal.status == "completed"
        local_event = next(
            event for event in result.audit if event.name == "vln.navigate.local"
        )
        assert local_event.arguments == {
            "instruction": local_instruction,
            "max_steps": 8,
        }
        exposed = {tool["name"]: tool for tool in responses.requests[0]["tools"]}
        assert "vln__navigate__local" in exposed
        assert "vln__navigate__task" not in exposed
        assert not any(
            name.endswith(("__start", "__status", "__cancel"))
            for name in exposed
        )
        assert "visible" in exposed["vln__navigate__local"]["description"]
        assert "must never be copied or paraphrased" in responses.requests[0][
            "instructions"
        ]

    run(scenario())


def test_normal_agent_requires_fresh_observation_and_local_instruction() -> None:
    async def scenario() -> None:
        goal_instruction = "Walk through the bedroom and dining room to the television."
        visible_instruction = (
            "Move through the visible doorway, stopping by the far wall of the hall."
        )
        responses = ScriptedResponses(
            [
                [
                    function_call(
                        "no-observation",
                        "vln__navigate__local",
                        {"instruction": visible_instruction, "max_steps": 8},
                    )
                ],
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        "copied-goal",
                        "vln__navigate__local",
                        {"instruction": goal_instruction, "max_steps": 8},
                    )
                ],
                [
                    function_call(
                        "valid-local",
                        "vln__navigate__local",
                        {"instruction": visible_instruction, "max_steps": 8},
                    )
                ],
                [
                    function_call(
                        "stale-observation",
                        "vln__navigate__local",
                        {"instruction": visible_instruction, "max_steps": 8},
                    )
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "policy checked"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe", "vln.navigate.local"),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", goal_instruction)
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("local-policy", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
                vln=DummyVLNNavigator(),
            ),
        )

        assert result.terminal.status == "completed"
        errors = [
            json.loads(responses.requests[index]["input"][-1]["output"])["error"]
            for index in (1, 3, 5)
        ]
        assert all(error["type"] == "AgentToolPolicyError" for error in errors)
        assert "fresh nav.observe" in errors[0]["message"]
        assert "not copy" in errors[1]["message"]
        assert "fresh nav.observe" in errors[2]["message"]
        assert [event.name for event in result.audit].count("vln.navigate.local") == 1

    run(scenario())


def test_normal_agent_rejects_nonlocal_vln_surfaces() -> None:
    for tool in (
        "vln.navigate.task",
        "vln.navigate.start",
        "vln.navigate.status",
        "vln.navigate.cancel",
    ):
        try:
            NormalAgent("test-model", ("nav.observe", tool))
        except ValueError as error:
            assert "only supports blocking local VLN navigation" in str(error)
        else:
            raise AssertionError(f"NormalAgent accepted nonlocal VLN tool: {tool}")

    try:
        NormalAgent("test-model", ("vln.navigate.local",))
    except ValueError as error:
        assert "requires nav.observe" in str(error)
    else:
        raise AssertionError("NormalAgent accepted local VLN without observation")


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

    for value in (True, 0, -1):
        with pytest.raises(ValueError, match="max_navigation_actions"):
            NormalAgent(
                "test-model",
                ("nav.observe",),
                max_navigation_actions=value,
            )


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


def test_normal_agent_requires_fresh_finish_and_accepted_goal_before_stop() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe-1", "nav__observe", {})],
                [
                    function_call(
                        "move", "nav__move__discrete", {"action": "forward"}
                    )
                ],
                [
                    function_call(
                        "stale-finish",
                        "nav__goal__finish",
                        {"status": "completed"},
                    )
                ],
                [function_call("observe-2", "nav__observe", {})],
                [
                    function_call(
                        "early-stop",
                        "nav__stop",
                        {"status": "completed", "reason": "too early"},
                    )
                ],
                [
                    function_call(
                        "finish", "nav__goal__finish", {"status": "completed"}
                    )
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "accepted"},
                    )
                ],
            ]
        )
        goal = NavGoal("goal", "move to the marker")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("finish-policy", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "nav.move.discrete", "nav.goal.finish"),
                    client=SimpleNamespace(responses=responses),
                ),
                DummyNavigationEnvironment((goal,), targets=(1,)),
            ),
        )

        assert result.terminal.status == "completed"
        assert [event.name for event in result.audit] == [
            "nav.observe",
            "nav.move.discrete",
            "nav.observe",
            "nav.goal.finish",
            "nav.stop",
        ]
        stale_error = json.loads(responses.requests[3]["input"][-1]["output"])
        stop_error = json.loads(responses.requests[5]["input"][-1]["output"])
        assert "fresh nav.observe" in stale_error["error"]["message"]
        assert "accepted final" in stop_error["error"]["message"]

    run(scenario())


def test_normal_agent_keeps_only_the_latest_observation_image() -> None:
    class VisualEnvironment(DummyNavigationEnvironment):
        async def _observe(self, actor, arguments):
            observation = await super()._observe(actor, arguments)
            observation["channels"]["rgb"] = np.zeros((12, 16, 3), dtype=np.uint8)
            return observation

    def image_count(request: dict[str, Any]) -> int:
        return sum(
            1
            for item in request["input"]
            if isinstance(item, dict)
            for part in (
                item.get("output", []) if isinstance(item.get("output"), list) else []
            )
            if isinstance(part, dict) and part.get("type") == "input_image"
        )

    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe-1", "nav__observe", {})],
                [
                    function_call(
                        "move", "nav__move__discrete", {"action": "forward"}
                    )
                ],
                [function_call("observe-2", "nav__observe", {})],
                [
                    function_call(
                        "stop", "nav__stop", {"status": "completed", "reason": "done"}
                    )
                ],
            ]
        )
        goal = NavGoal("goal", "inspect while moving")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("image-window", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "nav.move.discrete"),
                    client=SimpleNamespace(responses=responses),
                ),
                VisualEnvironment((goal,), targets=(1,)),
            ),
        )

        assert result.terminal.status == "completed"
        assert image_count(responses.requests[1]) == 1
        assert image_count(responses.requests[2]) == 0
        assert image_count(responses.requests[3]) == 1

    run(scenario())


def test_normal_agent_navigation_budget_rejects_batch_before_execution() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        f"too-many-{index}",
                        "nav__move__discrete",
                        {"action": "forward"},
                    )
                    for index in range(3)
                ],
                [
                    function_call(
                        f"move-{index}",
                        "nav__move__discrete",
                        {"action": "forward"},
                    )
                    for index in range(2)
                ],
                [
                    function_call(
                        "stop", "nav__stop", {"status": "completed", "reason": "done"}
                    )
                ],
            ]
        )
        goal = NavGoal("goal", "move within budget")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("action-budget", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "nav.move.discrete"),
                    max_navigation_actions=2,
                    client=SimpleNamespace(responses=responses),
                ),
                DummyNavigationEnvironment((goal,), targets=(2,)),
            ),
        )

        assert result.environment["position"] == 2
        assert [event.name for event in result.audit].count("nav.move.discrete") == 2
        rejected = responses.requests[2]["input"][-3:]
        assert all(
            "remaining navigation action budget" in json.loads(item["output"])[
                "error"
            ]["message"]
            for item in rejected
        )

    run(scenario())


def test_normal_agent_limits_repeated_object_candidate_calls() -> None:
    async def scenario() -> None:
        outputs: list[list[Any]] = []
        for index in range(4):
            outputs.extend(
                [
                    [function_call(f"observe-{index}", "nav__observe", {})],
                    [
                        function_call(
                            f"candidate-{index}",
                            "vln__navigate__local",
                            {
                                "instruction": (
                                    f"Approach the visible cabinet number {index} and "
                                    "stop in front of it."
                                ),
                                "max_steps": 4,
                            },
                        )
                    ],
                ]
            )
        outputs.append(
            [
                function_call(
                    "stop",
                    "nav__stop",
                    {"status": "failed", "reason": "retry guard checked"},
                )
            ]
        )
        responses = ScriptedResponses(outputs)
        goal = NavGoal(
            "goal", "Find the cabinet.", "object", {"category": "cabinet"}
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("candidate-retries", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "vln.navigate.local"),
                    client=SimpleNamespace(responses=responses),
                ),
                DummyNavigationEnvironment((goal,), targets=(0,)),
                vln=DummyVLNNavigator(),
            ),
        )

        assert result.terminal.status == "failed"
        assert [event.name for event in result.audit].count("vln.navigate.local") == 3
        instructions = " ".join(responses.requests[0]["instructions"].split())
        assert "appliance framed by cabinet panels" in instructions
        assert (
            "continue short calls while each one makes clear pose progress"
            in instructions
        )
        assert "never switch from one plausible instance" in instructions
        rejected = json.loads(responses.requests[8]["input"][-1]["output"])
        assert rejected["error"]["type"] == "AgentToolPolicyError"
        assert "retry limit" in rejected["error"]["message"]

    run(scenario())


def test_normal_agent_resets_object_candidate_limit_after_position_progress() -> None:
    async def scenario() -> None:
        outputs: list[list[Any]] = []
        for index in range(4):
            outputs.extend(
                [
                    [function_call(f"observe-{index}", "nav__observe", {})],
                    [
                        function_call(
                            f"candidate-{index}",
                            "vln__navigate__local",
                            {
                                "instruction": (
                                    "Approach the visible cabinet and stop in front "
                                    "of it."
                                ),
                                "max_steps": 1,
                            },
                        )
                    ],
                ]
            )
        outputs.append(
            [
                function_call(
                    "stop",
                    "nav__stop",
                    {"status": "failed", "reason": "progress guard checked"},
                )
            ]
        )
        responses = ScriptedResponses(outputs)
        goal = NavGoal(
            "goal", "Find the cabinet.", "object", {"category": "cabinet"}
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("candidate-progress", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "vln.navigate.local"),
                    client=SimpleNamespace(responses=responses),
                ),
                DummyNavigationEnvironment((goal,), targets=(10,)),
                vln=DummyVLNNavigator(local_max_steps=1),
            ),
        )

        assert result.terminal.status == "failed"
        assert [event.name for event in result.audit].count("vln.navigate.local") == 4
        assert result.environment["position"] == 4

    run(scenario())


def test_normal_agent_limits_object_local_navigation_to_eight_steps() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        "local",
                        "vln__navigate__local",
                        {
                            "instruction": (
                                "Go through the visible doorway and stop by its far "
                                "wall."
                            ),
                            "max_steps": 16,
                        },
                    )
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "failed", "reason": "step guard checked"},
                    )
                ],
            ]
        )
        goal = NavGoal(
            "goal", "Find the cabinet.", "object", {"category": "cabinet"}
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("object-step-limit", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "vln.navigate.local"),
                    client=SimpleNamespace(responses=responses),
                ),
                DummyNavigationEnvironment((goal,), targets=(10,)),
                vln=DummyVLNNavigator(local_max_steps=16),
            ),
        )

        assert result.terminal.status == "failed"
        assert "vln.navigate.local" not in [event.name for event in result.audit]
        rejected = json.loads(responses.requests[2]["input"][-1]["output"])
        assert rejected["error"]["type"] == "AgentToolPolicyError"
        assert "between 1 and 8" in rejected["error"]["message"]

        route_responses = ScriptedResponses(
            [
                [function_call("route-observe", "nav__observe", {})],
                [
                    function_call(
                        "route-local",
                        "vln__navigate__local",
                        {
                            "instruction": (
                                "Go through the visible doorway and stop by its far "
                                "wall."
                            ),
                            "max_steps": 16,
                        },
                    )
                ],
                [
                    function_call(
                        "route-stop",
                        "nav__stop",
                        {"status": "failed", "reason": "route cap checked"},
                    )
                ],
            ]
        )
        route_goal = NavGoal("route", "Go to the far room.", "language")
        route_result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("route-step-limit", route_goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "vln.navigate.local"),
                    client=SimpleNamespace(responses=route_responses),
                ),
                DummyNavigationEnvironment((route_goal,), targets=(20,)),
                vln=DummyVLNNavigator(local_max_steps=16),
            ),
        )

        assert route_result.environment["position"] == 16
        assert [event.name for event in route_result.audit].count(
            "vln.navigate.local"
        ) == 1

    run(scenario())


def test_normal_agent_limits_repeated_turns_at_one_position() -> None:
    async def scenario() -> None:
        outputs: list[list[Any]] = []
        for batch in range(7):
            outputs.append([function_call(f"observe-{batch}", "nav__observe", {})])
            outputs.append(
                [
                    function_call(
                        f"turn-{batch}-{index}",
                        "nav__move__discrete",
                        {"action": "turn_left"},
                    )
                    for index in range(4)
                ]
            )
        outputs.append(
            [
                function_call(
                    "stop",
                    "nav__stop",
                    {"status": "failed", "reason": "scan guard checked"},
                )
            ]
        )
        responses = ScriptedResponses(outputs)
        goal = NavGoal("goal", "inspect the room")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("stationary-scan-limit", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "nav.move.discrete"),
                    client=SimpleNamespace(responses=responses),
                ),
                DummyNavigationEnvironment((goal,), targets=(0,)),
            ),
        )

        assert result.terminal.status == "failed"
        assert [event.name for event in result.audit].count("nav.move.discrete") == 24
        rejected = responses.requests[-1]["input"][-4:]
        assert all(
            "stationary scan limit" in json.loads(item["output"])["error"][
                "message"
            ]
            for item in rejected
        )

    run(scenario())


def test_normal_agent_reports_remaining_navigation_budget() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        "move", "nav__move__discrete", {"action": "forward"}
                    )
                ],
                [
                    function_call(
                        "stop", "nav__stop", {"status": "completed", "reason": "done"}
                    )
                ],
            ]
        )
        goal = NavGoal("goal", "move once")
        await NavigationHarness(timeout_s=1).run_task(
            NavTask("budget-report", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "nav.move.discrete"),
                    max_navigation_actions=3,
                    client=SimpleNamespace(responses=responses),
                ),
                DummyNavigationEnvironment((goal,), targets=(1,)),
            ),
        )

        move_output = json.loads(responses.requests[2]["input"][-1]["output"])
        assert move_output["navigation_budget"] == {"used": 1, "remaining": 2}

    run(scenario())


def test_normal_agent_stops_move_batch_after_blocked_motion() -> None:
    class BlockingEnvironment(DummyNavigationEnvironment):
        async def _move(self, actor, arguments):
            result = await super()._move(actor, arguments)
            result["motion"] = {"blocked": True}
            return result

    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("observe", "nav__observe", {})],
                [
                    function_call(
                        f"move-{index}",
                        "nav__move__discrete",
                        {"action": "forward"},
                    )
                    for index in range(3)
                ],
                [
                    function_call(
                        "stop", "nav__stop", {"status": "completed", "reason": "done"}
                    )
                ],
            ]
        )
        goal = NavGoal("goal", "test blocked batch")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("blocked-batch", goal),
            NavigationStack(
                NormalAgent(
                    "test-model",
                    ("nav.observe", "nav.move.discrete"),
                    client=SimpleNamespace(responses=responses),
                ),
                BlockingEnvironment((goal,), targets=(3,)),
            ),
        )

        assert result.environment["action_count"] == 1
        assert [event.name for event in result.audit] == [
            "nav.observe",
            "nav.move.discrete",
            "nav.stop",
        ]
        batch_outputs = responses.requests[2]["input"][-3:]
        skipped = [json.loads(item["output"]) for item in batch_outputs[1:]]
        assert all(item["error"]["type"] == "ToolBatchSkipped" for item in skipped)

    run(scenario())


def test_normal_agent_uses_preferred_rgbd_observation_channels() -> None:
    class VisualEnvironment(DummyNavigationEnvironment):
        async def _observe(self, actor, arguments):
            observation = await super()._observe(actor, arguments)
            observation["channels"]["rgb"] = np.zeros(
                (480, 640, 3), dtype=np.uint8
            )
            observation["channels"]["third_rgb"] = np.full(
                (480, 640, 3), 200, dtype=np.uint8
            )
            observation["channels"]["depth"] = np.full(
                (480, 640, 1), 0.25, dtype=np.float32
            )
            observation["channels"]["third_depth"] = np.full(
                (480, 640, 1), 0.5, dtype=np.float32
            )
            observation["channels"]["depth_metadata"] = {
                "encoding": "linear_normalized",
                "minimum_m": 0.5,
                "maximum_m": 4.5,
            }
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
            observation_image_channel="third_rgb",
            observation_depth_channel="third_depth",
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
        assert model_output["sensor_summary"]["depth_channel"] == "third_depth"
        assert model_output["sensor_summary"]["depth"] == {
            "center": 0.5,
            "grid": [[0.5, 0.5, 0.5]] * 3,
            "minimum": 0.5,
            "maximum": 0.5,
            "lower_is_nearer": True,
            "meters": {
                "center": 2.5,
                "grid": [[2.5, 2.5, 2.5]] * 3,
                "sensor_range": [0.5, 4.5],
            },
        }
        assert content[1]["type"] == "input_image"
        assert content[1]["detail"] == "high"
        assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
        encoded = content[1]["image_url"].partition(",")[2]
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert float(np.asarray(image).mean()) > 190

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

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from domain import DomainRuntime, DomainSpec, ModuleSpec, NavigationEpisode
from domain.modules import EnvironmentModule


MEMORY = {
    "current_place": "hallway",
    "completed_route": "left the start room",
    "next_route_step": "approach the endpoint",
    "decision": "the visible hall is open",
}
EVIDENCE = {
    "route_complete": True,
    "endpoint_visible": True,
    "close_enough": True,
    "summary": (
        "left the start room, followed the hall, and approached the visible endpoint"
    ),
}


class VisualEnvironment(EnvironmentModule):
    def __init__(self) -> None:
        super().__init__()
        self.actions: list[str] = []
        self.pose = [0.0, 0.0]

    def mount(self) -> None:
        self.expose(
            "env.observe",
            self.observe,
            description="Return the current RGB-D observation.",
            parameters={"type": "object", "additionalProperties": False},
        )
        self.expose(
            "env.step",
            self.step,
            description="Execute one navigation action.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"enum": ["forward", "turn_left", "turn_right"]}
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            mutates=True,
        )

    def observe(self) -> dict[str, Any]:
        return {
            "observation_id": len(self.actions) + 1,
            "channels": {
                "rgb": np.full((12, 16, 3), 127, dtype=np.uint8),
                "depth": np.full((12, 16), 2.5, dtype=np.float32),
                "gps": np.asarray(self.pose, dtype=np.float32),
                "compass": np.asarray([0.0], dtype=np.float32),
            },
            "pose": list(self.pose),
            "action_count": len(self.actions),
        }

    def step(self, action: str) -> dict[str, Any]:
        self.actions.append(action)
        if action == "forward":
            self.pose[1] += 0.25
        return {
            "accepted": True,
            "action": action,
            "action_count": len(self.actions),
            "terminal": False,
            "pose": list(self.pose),
        }

    def stop(self, reason: str) -> None:
        del reason

    def result(self) -> dict[str, Any]:
        return {"success": True, "actions": list(self.actions)}


class FirstForwardBlockedEnvironment(VisualEnvironment):
    def step(self, action: str) -> dict[str, Any]:
        if not self.actions and action == "forward":
            self.actions.append(action)
            return {
                "accepted": True,
                "action": action,
                "action_count": len(self.actions),
                "terminal": False,
                "pose": list(self.pose),
            }
        return super().step(action)


class ScriptedResponses:
    def __init__(self, outputs: list[list[Any]]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append(request)
        output = self.outputs.pop(0)
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            model_dump=lambda **_: {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
        )
        return SimpleNamespace(
            id=f"response-{len(self.requests)}",
            usage=usage,
            output=output,
        )


def function_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def reasoning(text: str) -> Any:
    return SimpleNamespace(
        type="reasoning",
        summary=[SimpleNamespace(text=text)],
    )


def act(
    call_id: str,
    actions: list[str],
    *,
    memory: dict[str, str] | None = None,
) -> Any:
    return function_call(
        call_id,
        "nav_act",
        {"actions": actions, "navigation_memory": memory or MEMORY},
    )


def stop(call_id: str, *, confirm: bool) -> Any:
    return function_call(
        call_id,
        "nav_stop",
        {
            "status": "completed",
            "reason": "destination reached",
            "actor": "agent",
            "confirm": confirm,
            "arrival_evidence": EVIDENCE,
        },
    )


def agent_spec(client: Any, **params: Any) -> ModuleSpec:
    return ModuleSpec(
        "normal_agent",
        "modules.normal_agent:NormalAgent",
        {
            "model": "test-model",
            "tools": ["env.observe", "env.step"],
            "max_iterations": 12,
            "model_retries": 0,
            "client": client,
            **params,
        },
    )


def domain_spec(client: Any, **params: Any) -> DomainSpec:
    return DomainSpec(
        ModuleSpec("env", "envs.replay:ReplayEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        (agent_spec(client, **params),),
        timeout_s=2,
        shutdown_timeout_s=1,
    )


def test_normal_agent_preserves_nexus_native_atomic_loop(tmp_path) -> None:
    responses = ScriptedResponses(
        [
            [act("act", ["move_forward", "move_forward"])],
            [stop("candidate", confirm=False)],
            [stop("confirm", confirm=True)],
        ]
    )
    client = SimpleNamespace(responses=responses)
    episode = NavigationEpisode(
        "atomic-agent",
        {"type": "instruction", "instruction": "Move forward twice."},
        truth={"expert_actions": ["forward", "forward"]},
    )

    result = DomainRuntime().run(
        episode,
        domain_spec(client),
        tmp_path,
        domain_id="atomic-agent",
    )

    assert result.terminal.status == "completed"
    assert result.environment["actions"] == ["forward", "forward"]
    assert not result.errors
    first = responses.requests[0]
    assert first["parallel_tool_calls"] is False
    assert first["tool_choice"] == "required"
    assert [tool["name"] for tool in first["tools"]] == [
        "nav_act",
        "nav_observe",
        "nav_stop",
    ]
    assert {tool["name"] for tool in first["tools"]} == {
        "nav_observe",
        "nav_act",
        "nav_stop",
    }
    action_tool = next(tool for tool in first["tools"] if tool["name"] == "nav_act")
    assert action_tool["parameters"]["properties"]["actions"]["maxItems"] == 4
    assert "navigation_memory" in action_tool["parameters"]["required"]
    stop_tool = next(tool for tool in first["tools"] if tool["name"] == "nav_stop")
    assert {"status", "reason", "actor", "confirm", "arrival_evidence"} <= set(
        stop_tool["parameters"]["required"]
    )
    first_state = json.loads(first["input"][0]["content"][0]["text"].split("\n", 1)[1])
    public_observation = first_state["observation"]
    assert isinstance(public_observation["time_unix"], float)
    assert "frame" not in public_observation["pose"]
    calls = [
        json.loads(line)
        for line in (tmp_path / "atomic-agent/calls.jsonl").read_text().splitlines()
    ]
    assert [item["name"] for item in calls if item["operation"] == "call"] == [
        "env.observe",
        "env.step",
        "env.step",
        "env.observe",
        "env.stop",
    ]


def test_normal_agent_executes_full_batch_after_blocked_action(tmp_path) -> None:
    responses = ScriptedResponses(
        [
            [act("act", ["move_forward", "move_forward", "move_forward"])],
            [stop("candidate", confirm=False)],
            [stop("confirm", confirm=True)],
        ]
    )
    client = SimpleNamespace(responses=responses)
    episode = NavigationEpisode(
        "blocked-batch",
        {"type": "instruction", "instruction": "Continue after a blocked step."},
    )
    spec = DomainSpec(
        ModuleSpec("env", f"{__name__}:FirstForwardBlockedEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        (agent_spec(client),),
        timeout_s=2,
        shutdown_timeout_s=1,
    )

    result = DomainRuntime().run(
        episode,
        spec,
        tmp_path,
        domain_id="blocked-batch",
    )

    assert result.environment["actions"] == ["forward", "forward", "forward"]
    assert not result.errors
    history = json.loads(
        (
            tmp_path / "blocked-batch/workspace/modules/normal_agent/model/history.json"
        ).read_text()
    )
    assert history["summary"]["action_summary"]["total"] == 3
    assert history["turns"][0]["blocked_actions"] == ["move_forward"]


def test_normal_agent_uniformly_samples_images_and_keeps_model_progress(
    tmp_path,
) -> None:
    outputs: list[list[Any]] = [
        [
            reasoning(f"view summary {index}"),
            act(
                f"move-{index}",
                ["move_forward"],
                memory={
                    **MEMORY,
                    "completed_route": f"instruction steps 1 through {index}",
                },
            ),
        ]
        for index in range(4)
    ]
    outputs.extend(
        [[stop("candidate", confirm=False)], [stop("confirm", confirm=True)]]
    )
    responses = ScriptedResponses(outputs)
    client = SimpleNamespace(responses=responses)
    episode = NavigationEpisode(
        "memory-agent",
        {"type": "instruction", "instruction": "Walk down the hall."},
        truth={"expert_actions": ["forward"] * 4},
    )

    spec = DomainSpec(
        ModuleSpec("env", f"{__name__}:VisualEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        (agent_spec(client, max_iterations=6, image_memory_turns=2),),
        timeout_s=2,
        shutdown_timeout_s=1,
    )
    result = DomainRuntime().run(
        episode,
        spec,
        tmp_path,
        domain_id="memory-agent",
    )

    assert result.terminal.status == "completed"
    final_input = responses.requests[-1]["input"]
    image_count = sum(
        1
        for item in final_input
        if isinstance(item, dict)
        for content in item.get("content", [])
        if content.get("type") == "input_image"
    )
    text = "\n".join(
        content.get("text", "")
        for item in final_input
        if isinstance(item, dict)
        for content in item.get("content", [])
    )
    assert image_count == 2
    sampled_steps = []
    for item in final_input:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not any(part.get("type") == "input_image" for part in content):
            continue
        state_text = next(
            part["text"]
            for part in content
            if part.get("type") == "input_text"
            and part.get("text", "").startswith("Current navigation state:\n")
        )
        sampled_steps.append(
            json.loads(state_text.split("\n", 1)[1])["observation"]["step"]
        )
    assert sampled_steps == [0, 4]
    assert "uniformly sampled" in text
    assert "view summary 0" in text
    assert "instruction steps 1 through 3" in text
    history = json.loads(
        (
            tmp_path / "memory-agent/workspace/modules/normal_agent/model/history.json"
        ).read_text()
    )
    assert history["summary"]["total_turns"] == 6
    assert history["summary"]["model_instruction_progress"] == {
        "current_place": "hallway",
        "completed_instruction_steps": EVIDENCE["summary"],
        "next_instruction_step": "none; model declared route complete",
        "last_decision": "destination reached",
    }
    assert len(history["turns"]) == 6


def test_normal_agent_rejects_early_stop_and_requires_confirmation(tmp_path) -> None:
    responses = ScriptedResponses(
        [
            [stop("early", confirm=False)],
            [act("approach", ["move_forward"] * 4)],
            [stop("candidate", confirm=False)],
            [stop("confirm", confirm=True)],
        ]
    )
    client = SimpleNamespace(responses=responses)
    episode = NavigationEpisode(
        "guarded-agent",
        {"type": "instruction", "instruction": "Walk to the landmark."},
        truth={"expert_actions": ["forward"] * 4},
    )

    result = DomainRuntime().run(
        episode,
        domain_spec(client, max_iterations=4, minimum_travel_m=1.0),
        tmp_path,
        domain_id="guarded-agent",
    )

    assert result.terminal.status == "completed"
    history = json.loads(
        (
            tmp_path / "guarded-agent/workspace/modules/normal_agent/model/history.json"
        ).read_text()
    )
    assert history["turns"][0]["error"].startswith("premature stop rejected")
    assert history["turns"][-2]["error"].startswith("arrival candidate recorded")


def test_normal_agent_sends_rgbd_and_meter_depth_grid(tmp_path) -> None:
    responses = ScriptedResponses(
        [[stop("candidate", confirm=False)], [stop("confirm", confirm=True)]]
    )
    client = SimpleNamespace(responses=responses)
    spec = DomainSpec(
        ModuleSpec("env", f"{__name__}:VisualEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        (agent_spec(client, max_iterations=2),),
        timeout_s=2,
        shutdown_timeout_s=1,
    )

    result = DomainRuntime().run(
        NavigationEpisode(
            "visual-agent",
            {"type": "instruction", "instruction": "Stop here."},
        ),
        spec,
        tmp_path,
        domain_id="visual-agent",
    )

    assert result.terminal.status == "completed"
    current = responses.requests[0]["input"][-1]
    assert [part["type"] for part in current["content"]] == [
        "input_text",
        "input_image",
    ]
    assert current["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    state = json.loads(current["content"][0]["text"].split("\n", 1)[1])
    assert state["observation"]["depth_grid_m"] == [
        [2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5],
        [2.5, 2.5, 2.5],
    ]
    trace = (
        tmp_path / "visual-agent/workspace/modules/normal_agent/model/trace.jsonl"
    ).read_text()
    assert "data:image/jpeg;base64," not in trace

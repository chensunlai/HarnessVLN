from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from domain import DomainRuntime, DomainSpec, ModuleSpec, NavigationEpisode
from domain.modules import EnvironmentModule
from modules.agent_vln.module import _memory_indices, _tools


MODULES = (
    ModuleSpec("agent_vln", "modules.agent_vln:AgentVLNModule"),
    ModuleSpec("agent_objnav", "modules.agent_objnav:AgentObjNavModule"),
    ModuleSpec("agent_local", "modules.agent_local:AgentLocalModule"),
    ModuleSpec("agent_desc", "modules.agent_desc:AgentDescModule"),
    ModuleSpec("master_agent", "modules.master_agent:MasterAgent"),
)


PROGRESS = {
    "current_place": "hallway",
    "completed_steps": ["walked through the doorway"],
    "next_step": "stop beside the table",
    "decision": "the endpoint is now close",
}


class LocalVisualEnvironment(EnvironmentModule):
    def __init__(self) -> None:
        super().__init__()
        self.actions: list[str] = []
        self.position = [0.0, 0.0, 0.0]

    def mount(self) -> None:
        self.expose(
            "env.observe",
            self.observe,
            description="Return RGB-D and pose.",
            parameters={"type": "object", "additionalProperties": False},
        )
        self.expose(
            "env.step",
            self.step,
            description="Execute one atomic navigation action.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "enum": ["forward", "turn_left", "turn_right"]
                    }
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
            },
            "pose": {"position": list(self.position), "heading_degrees": 0.0},
            "action_count": len(self.actions),
        }

    def step(self, action: str) -> dict[str, Any]:
        self.actions.append(action)
        if action == "forward":
            self.position[2] += 0.25
        return {
            "accepted": True,
            "action": action,
            "terminal": False,
            "pose": {"position": list(self.position), "heading_degrees": 0.0},
        }

    def stop(self, reason: str) -> None:
        del reason

    def result(self) -> dict[str, Any]:
        return {"success": True, "actions": list(self.actions)}


class ScriptedResponses:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(
            id=f"response-{len(self.requests)}",
            output=[self.outputs.pop(0)],
            usage=SimpleNamespace(
                model_dump=lambda **_: {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                }
            ),
        )


def _call(name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id=name,
        name=name,
        arguments=json.dumps(arguments),
    )


def test_agent_vln_memory_and_exhausted_budget_tools() -> None:
    assert _memory_indices(10, 6, 3) == [0, 3, 6, 7, 8, 9]
    assert [tool["name"] for tool in _tools(False, action_available=False)] == [
        "agent_vln_finish"
    ]


@pytest.mark.parametrize(
    ("instruction", "expected_function"),
    [
        (
            {"type": "instruction", "instruction": "Walk through the visible door."},
            "agent_vln.run",
        ),
        (
            {"type": "target_text", "instruction": "chair"},
            "agent_objnav.run",
        ),
        (
            {
                "type": "target_pose",
                "target_pose": {
                    "frame": "agent",
                    "position": [1.0, 0.0],
                    "heading_degrees": 180.0,
                },
            },
            "agent_local.run",
        ),
        ({"type": "describe"}, "agent_desc.run"),
    ],
)
def test_master_agent_dispatches_to_each_subagent(
    tmp_path, instruction, expected_function
) -> None:
    episode = NavigationEpisode("subagents", instruction)
    spec = DomainSpec(
        ModuleSpec("env", "envs.replay:ReplayEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        MODULES,
        timeout_s=2,
        shutdown_timeout_s=1,
    )

    result = DomainRuntime().run(episode, spec, tmp_path, domain_id=expected_function)

    assert result.terminal.status == "incomplete"
    assert not result.errors
    dispatch_path = (
        tmp_path
        / expected_function
        / "workspace/modules/master_agent/dispatch.json"
    )
    dispatch = json.loads(dispatch_path.read_text())
    assert dispatch["function"] == expected_function
    assert dispatch["result"]["status"] == "refused"

    register = json.loads(
        (tmp_path / expected_function / "register.json").read_text()
    )
    assert {
        "agent_vln.run",
        "agent_objnav.run",
        "agent_local.run",
        "agent_desc.run",
    } <= set(register["functions"])


def test_agent_vln_runs_native_tool_loop_and_returns_to_master(tmp_path) -> None:
    completed = {
        **PROGRESS,
        "completed_steps": [
            "walked through the doorway",
            "stopped beside the table",
        ],
        "next_step": "none",
    }
    responses = ScriptedResponses(
        [
            _call(
                "agent_vln_act",
                {"actions": ["forward", "forward"], "progress": PROGRESS},
            ),
            _call(
                "agent_vln_finish",
                {
                    "status": "completed",
                    "reason": "all route clauses are complete",
                    "confirm": False,
                    "progress": completed,
                    "endpoint": {
                        "visible": True,
                        "near": True,
                        "description": "the table is at standing distance",
                    },
                },
            ),
            _call(
                "agent_vln_finish",
                {
                    "status": "completed",
                    "reason": "arrival confirmed",
                    "confirm": True,
                    "progress": completed,
                    "endpoint": {
                        "visible": True,
                        "near": True,
                        "description": "the table remains at standing distance",
                    },
                },
            ),
        ]
    )
    client = SimpleNamespace(responses=responses)
    modules = (
        ModuleSpec(
            "agent_vln",
            "modules.agent_vln:AgentVLNModule",
            {
                "model": "test-model",
                "minimum_travel_m": 0.505,
                "image_memory": 2,
                "recent_images": 2,
                "model_retries": 0,
                "client": client,
            },
        ),
        ModuleSpec("master_agent", "modules.master_agent:MasterAgent"),
    )
    spec = DomainSpec(
        ModuleSpec("env", f"{__name__}:LocalVisualEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        modules,
        timeout_s=2,
        shutdown_timeout_s=1,
    )
    episode = NavigationEpisode(
        "agent-vln-loop",
        {
            "type": "instruction",
            "instruction": "Walk through the doorway and stop beside the table.",
        },
    )

    result = DomainRuntime().run(
        episode, spec, tmp_path, domain_id="agent-vln-loop"
    )

    assert result.terminal.status == "completed"
    assert result.environment["actions"] == ["forward", "forward"]
    assert not result.errors
    request = responses.requests[0]
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False
    assert [tool["name"] for tool in request["tools"]] == [
        "agent_vln_act",
        "agent_vln_finish",
    ]
    assert (
        request["tools"][0]["parameters"]["properties"]["actions"]["maxItems"]
        == 4
    )
    assert responses.requests[-1]["tools"][1]["parameters"]["properties"][
        "confirm"
    ]["enum"] == [True]
    final_content = responses.requests[-1]["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in final_content) == 2
    trace = (
        tmp_path
        / "agent-vln-loop/workspace/modules/agent_vln/trace.jsonl"
    ).read_text()
    assert "data:image/jpeg;base64," not in trace
    history = json.loads(
        (
            tmp_path
            / "agent-vln-loop/workspace/modules/agent_vln/history.json"
        ).read_text()
    )
    assert history["actions"] == ["forward", "forward"]
    assert history["travelled_m"] == 0.5

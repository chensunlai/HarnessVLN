from __future__ import annotations

import json

import pytest

from domain import DomainRuntime, DomainSpec, ModuleSpec, NavigationEpisode


MODULES = (
    ModuleSpec("agent_vln", "modules.agent_vln:AgentVLNModule"),
    ModuleSpec("agent_objnav", "modules.agent_objnav:AgentObjNavModule"),
    ModuleSpec("agent_local", "modules.agent_local:AgentLocalModule"),
    ModuleSpec("agent_desc", "modules.agent_desc:AgentDescModule"),
    ModuleSpec("master_agent", "modules.master_agent:MasterAgent"),
)


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

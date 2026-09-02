from __future__ import annotations

import json

from domain import DomainRuntime, DomainSpec, ModuleSpec, NavigationEpisode, WorkspaceSpec


def test_expert_domain_writes_complete_episode(tmp_path) -> None:
    episode = NavigationEpisode(
        "episode-1",
        {"type": "instruction", "instruction": "Follow the route."},
        truth={"expert_actions": ["forward", "left", "forward"]},
    )
    spec = DomainSpec(
        ModuleSpec("env", "envs.replay:ReplayEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        (ModuleSpec("expert", "modules.expert:ExpertTrajectoryModule"),),
        WorkspaceSpec(),
        timeout_s=2,
        shutdown_timeout_s=1,
    )
    result = DomainRuntime().run(episode, spec, tmp_path, domain_id="episode-1")

    assert result.terminal.status == "completed"
    assert result.metrics == {"success": 1.0, "path_efficiency": 1.0}
    assert not result.errors
    root = tmp_path / "episode-1"
    assert (root / "workspace/modules/expert/trajectory.json").is_file()
    calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
    assert [item["name"] for item in calls].count("env.step") == 3
    assert calls[-1]["name"] == "env.stop"


def test_arbitrary_modules_call_each_other_across_threads(tmp_path) -> None:
    episode = NavigationEpisode(
        "services", {"type": "instruction", "instruction": "Exercise module services."}
    )
    spec = DomainSpec(
        ModuleSpec("env", "envs.replay:ReplayEnvironment"),
        ModuleSpec("metric", "metrics.navigation:NavigationMetric"),
        (
            ModuleSpec("echo", "tests.fixtures.service_modules:EchoService"),
            ModuleSpec("driver", "tests.fixtures.service_modules:ServiceDriver"),
        ),
        timeout_s=2,
        shutdown_timeout_s=1,
    )

    result = DomainRuntime().run(episode, spec, tmp_path, domain_id="services")

    assert result.terminal.status == "completed"
    echo = json.loads(
        (tmp_path / "services/workspace/modules/driver/echo.json").read_text()
    )
    assert echo == {"value": "ready", "thread": "domain-echo"}
    assert not result.errors

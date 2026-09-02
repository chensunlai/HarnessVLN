from __future__ import annotations

import json

from domain import DomainRuntime, DomainSpec, ModuleSpec, NavigationEpisode


def test_async_native_session_is_normalized(tmp_path) -> None:
    episode = NavigationEpisode(
        "async-session",
        {"type": "instruction", "instruction": "Move forward twice."},
        truth={"expert_actions": ["forward", "forward"]},
    )
    environment = ModuleSpec(
        "env",
        "envs.session:SessionEnvironment",
        {
            "session_factory": "tests.fixtures.fake_session:create",
            "native_actions": {"forward": "native_forward"},
            "terminal_action": "native_stop",
            "observation_channels": ["rgb", "pose"],
        },
    )
    metric = ModuleSpec(
        "metric",
        "metrics.navigation:NavigationMetric",
        {"fields": {"success": "success", "distance": "distance_to_goal"}},
    )
    spec = DomainSpec(
        environment,
        metric,
        (ModuleSpec("expert", "modules.expert:ExpertTrajectoryModule"),),
        timeout_s=2,
        shutdown_timeout_s=1,
    )

    result = DomainRuntime().run(episode, spec, tmp_path, domain_id="async-session")

    assert result.terminal.status == "completed"
    assert result.environment["success"] is True
    assert result.environment["final_pose"] == [2, 0, 0]
    assert result.metrics == {"success": 1.0, "distance": 0.0}
    assert not result.errors


def test_multi_goal_session_stays_in_one_domain(tmp_path) -> None:
    episode = NavigationEpisode(
        "multi-goal",
        {
            "type": "goals",
            "goals": [
                {"type": "target_text", "instruction": "chair"},
                {"type": "target_img", "image": "goal.png"},
            ]
        },
    )
    environment = ModuleSpec(
        "env",
        "envs.session:SessionEnvironment",
        {
            "session_factory": "tests.fixtures.fake_session:create_multi_goal",
            "native_actions": {"forward": "forward"},
            "goal_action": "finish_goal",
            "terminal_action": None,
        },
    )
    metric = ModuleSpec(
        "metric",
        "metrics.navigation:NavigationMetric",
        {"fields": {"goals_completed": "goals_completed", "success": "success"}},
    )
    spec = DomainSpec(
        environment,
        metric,
        (ModuleSpec("driver", "tests.fixtures.goal_driver:GoalDriver"),),
        timeout_s=2,
        shutdown_timeout_s=1,
    )

    result = DomainRuntime().run(episode, spec, tmp_path, domain_id="multi-goal")

    assert result.terminal.status == "environment_terminal"
    assert result.environment["goal_index"] == 2
    assert result.metrics == {"goals_completed": 2.0, "success": 1.0}
    assert not result.errors
    goals = json.loads(
        (tmp_path / "multi-goal/workspace/modules/driver/goals.json").read_text()
    )
    assert goals["initial"]["instruction"] == {
        "type": "target_text",
        "instruction": "chair",
    }
    assert goals["next"]["instruction"] == {
        "type": "target_img",
        "image": "goal.png",
    }


def test_isaac_high_level_action_waits_for_native_ticks(tmp_path) -> None:
    episode = NavigationEpisode(
        "isaac-ticks",
        {"type": "instruction", "instruction": "Move forward."},
        truth={"expert_actions": ["forward"]},
    )
    spec = DomainSpec(
        ModuleSpec(
            "env",
            "envs.isaac:IsaacEnvironment",
            {
                "runtime_factory": "tests.fixtures.fake_session:create_isaac",
                "max_native_ticks_per_action": 5,
            },
        ),
        ModuleSpec(
            "metric",
            "metrics.navigation:NavigationMetric",
            {"fields": {"success": "success", "spl": "spl"}},
        ),
        (ModuleSpec("expert", "modules.expert:ExpertTrajectoryModule"),),
        timeout_s=2,
        shutdown_timeout_s=1,
    )

    result = DomainRuntime().run(episode, spec, tmp_path, domain_id="isaac-ticks")

    assert result.terminal.status == "completed"
    assert result.environment["native_tick_count"] == 9
    assert result.metrics == {"success": 1.0, "spl": 1.0}
    assert not result.errors

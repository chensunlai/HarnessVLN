from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np

from benches.base import BenchmarkCase
from envs.habitat import HabitatEnvironment, _rewrite_prefix, _same_scene
from envs.habitat.environment import _quiet_habitat_loggers
from harness import NavigationHarness, NavigationStack
from harness.output import RunOutput
from harness.requirements import check_navigation_requirements
from schemas import NavGoal, NavTask
from vln import StreamVLNNavigator


class FakeHabitatSession:
    def __init__(self) -> None:
        self.reset_count = 0
        self.actions: list[object] = []
        self.closed = False
        self.episode_over = False

    def reset(self):
        self.reset_count += 1
        return self._observation()

    def step(self, action):
        self.actions.append(action)
        return self._observation()

    def get_metrics(self):
        return {"native_step_count": len(self.actions)}

    def close(self):
        self.closed = True

    @staticmethod
    def _observation():
        return {
            "rgb": "rgb-frame",
            "depth": "depth-frame",
            "gps": [1.0, 2.0],
            "compass": [0.5],
            "private_metric": 99,
        }


def test_habitat_quieting_only_changes_module_loggers(monkeypatch) -> None:
    habitat_logger = logging.Logger("fixture.habitat", logging.INFO)
    simulator_logger = logging.Logger("fixture.habitat_sim", logging.INFO)
    habitat = SimpleNamespace(logger=habitat_logger)
    habitat_sim = SimpleNamespace(
        logging=SimpleNamespace(logger=simulator_logger)
    )
    environment = dict(os.environ)
    monkeypatch.setitem(sys.modules, "habitat_sim", habitat_sim)

    _quiet_habitat_loggers(habitat)

    assert habitat_logger.level == logging.ERROR
    assert simulator_logger.level == logging.ERROR
    assert dict(os.environ) == environment


def compound_case() -> BenchmarkCase:
    first = NavGoal("goal:0", "Find the chair.", "object")
    second = NavGoal("goal:1", "Find the lamp.", "description")
    task = NavTask("goat:fixture", first, scene_id="fixture.glb")
    return BenchmarkCase(
        "goat:fixture",
        task,
        {"goal_stream": (first, second)},
    )


def test_habitat_compound_task_keeps_one_native_session() -> None:
    class Agent:
        required_tools = frozenset(
            {"nav.observe", "nav.move.discrete", "nav.goal.finish"}
        )

        async def run(self, context):
            first = await context.nav.observe()
            assert first["channels"]["rgb"] == "rgb-frame"
            assert np.array_equal(first["channels"]["camera_intrinsics"], np.eye(4))
            assert first["channels"]["pose"]["frame"] == "habitat_episode"
            assert "private_metric" not in first["channels"]

            await context.nav.move_discrete("forward")
            transition = await context.nav.finish_goal("completed")
            assert transition["done"] is False
            assert transition["goal"]["goal_id"] == "goal:1"

            second = await context.nav.observe()
            assert second["extras"]["goal_id"] == "goal:1"
            await context.nav.move_discrete("turn_left")
            assert await context.nav.finish_goal("completed") == {
                "done": True,
                "accepted": True,
            }
            await context.nav.stop("completed")

    async def scenario():
        fixture = compound_case()
        session = FakeHabitatSession()
        environment = HabitatEnvironment(
            fixture.environment_episode,
            native_factory=lambda _: session,
            goal_finish_action="SUBTASK_STOP",
            static_channels={"camera_intrinsics": np.eye(4)},
            camera={"height": 480, "width": 640, "hfov_deg": 79},
        )
        assert environment.profile.observation_channels == frozenset(
            {"rgb", "depth", "gps", "compass", "pose", "camera_intrinsics"}
        )
        check_navigation_requirements(
            "StreamVLNNavigator",
            StreamVLNNavigator.requirements,
            environment.profile,
        )

        result = await NavigationHarness(timeout_s=1).run_task(
            fixture.task,
            NavigationStack(Agent(), environment),
        )

        assert session.reset_count == 1
        assert session.actions == [1, "SUBTASK_STOP", 2, "SUBTASK_STOP"]
        assert session.closed
        assert result.environment["goal_index"] == 1
        assert result.environment["goal_count"] == 2
        assert result.environment["native_step_count"] == 4
        assert result.environment["final_pose"] == {
            "frame": "habitat_episode",
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "yaw": 0.5,
        }

    asyncio.run(scenario())


def test_habitat_profile_does_not_claim_undeclared_channels() -> None:
    environment = HabitatEnvironment(
        compound_case().environment_episode,
        native_factory=lambda _: FakeHabitatSession(),
        observation_channels=("rgb",),
        expose_pose=False,
    )

    assert environment.profile.observation_channels == frozenset({"rgb"})


def test_habitat_environment_owns_main_camera_video_submission(tmp_path) -> None:
    class VideoSession(FakeHabitatSession):
        def _observation(self):
            value = len(self.actions) * 40
            return {
                "rgb": np.full((4, 6, 3), value, dtype=np.uint8),
                "depth": np.zeros((4, 6), dtype=np.float32),
                "gps": [1.0, 2.0],
                "compass": [0.5],
            }

    class Agent:
        required_tools = frozenset({"nav.move.discrete"})

        async def run(self, context):
            await context.nav.move_discrete("forward")
            await context.nav.stop("completed")

    async def scenario():
        fixture = compound_case()
        run_output = RunOutput(
            {
                "root": str(tmp_path),
                "run_id": "habitat-video",
                "video": {"fps": 5},
            },
            resolved_config={},
            config_sources=(),
            config_digest="d" * 64,
            provenance={},
        )
        bench_output = run_output.benchmark(0, "habitat", "fixture")
        episode_output = bench_output.episode(0, fixture.case_id, fixture.output_record())
        environment = HabitatEnvironment(
            fixture.environment_episode,
            native_factory=lambda _: VideoSession(),
        )

        result = await NavigationHarness(timeout_s=1).run_task(
            fixture.task,
            NavigationStack(Agent(), environment),
            output=episode_output,
        )
        assert episode_output.finish({"case_id": fixture.case_id}) == ()

        document = json.loads(
            (episode_output.path / "environment.json").read_text()
        )
        stream = document["streams"]["main_camera"]
        assert stream["status"] == "saved"
        assert stream["frame_count"] == 2
        assert (run_output.path / stream["path"]).is_file()
        assert document["result"] == result.environment

    asyncio.run(scenario())


def test_habitat_explicit_noop_is_a_logical_action_without_native_stop() -> None:
    async def scenario():
        fixture = compound_case()
        session = FakeHabitatSession()
        environment = HabitatEnvironment(
            fixture.environment_episode,
            native_factory=lambda _: session,
            native_actions={"stand_still": None, "forward": 1},
        )

        await environment.start(fixture.task)
        result = await environment._move("vln", {"action": "stand_still"})
        await environment.stop("done")

        assert result == {
            "action": "stand_still",
            "action_count": 1,
            "goal_action_count": 1,
            "native_terminal": False,
            "motion": {
                "translation_m": 0.0,
                "rotation_deg": 0.0,
                "blocked": False,
                "pose": {
                    "frame": "habitat_episode",
                    "x": 1.0,
                    "y": 2.0,
                    "z": 0.0,
                    "yaw": 0.5,
                },
            },
        }
        assert session.actions == []
        assert environment.result()["action_count"] == 1

    asyncio.run(scenario())


def test_habitat_atomic_motion_reports_blocked_forward() -> None:
    async def scenario():
        fixture = compound_case()
        environment = HabitatEnvironment(
            fixture.environment_episode,
            native_factory=lambda _: FakeHabitatSession(),
        )

        await environment.start(fixture.task)
        result = await environment._move("agent", {"action": "forward"})
        await environment.stop("done")

        assert result["motion"]["translation_m"] == 0.0
        assert result["motion"]["rotation_deg"] == 0.0
        assert result["motion"]["blocked"] is True

    asyncio.run(scenario())


def test_habitat_adapter_tracks_oracle_success_from_minimum_distance() -> None:
    class DistanceSession(FakeHabitatSession):
        distances = (4.0, 2.5, 3.5)

        def get_metrics(self):
            return {
                "distance_to_goal": self.distances[min(len(self.actions), 2)],
                "success": 0.0,
                "spl": 0.0,
            }

    async def scenario():
        fixture = compound_case()
        session = DistanceSession()
        environment = HabitatEnvironment(
            fixture.environment_episode,
            native_factory=lambda _: session,
            oracle_success_distance=3.0,
        )

        await environment.start(fixture.task)
        assert environment.result()["oracle_success"] == 0.0
        await environment._move("agent", {"action": "forward"})
        assert environment.result()["oracle_success"] == 1.0
        await environment._move("agent", {"action": "forward"})
        assert environment.result()["oracle_success"] == 1.0
        await environment.stop("done")

    asyncio.run(scenario())


def test_habitat_scene_matching_accepts_dataset_resolved_paths() -> None:
    relative = "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
    absolute = f"/datasets/scene_datasets/{relative}"

    assert _same_scene(absolute, relative)
    assert _same_scene(relative, absolute)
    assert not _same_scene(absolute, "mp3d/other/other.glb")


def test_habitat_scene_prefix_rewrite_is_explicit_and_ordered() -> None:
    value = "data/scene_datasets/hm3d_v0.2/val/scene/scene.basis.glb"
    assert _rewrite_prefix(
        value,
        {"data/scene_datasets/hm3d_v0.2/": "data/scene_datasets/hm3d/"},
    ) == "data/scene_datasets/hm3d/val/scene/scene.basis.glb"
    assert _rewrite_prefix(value, {"other/": "replacement/"}) == value

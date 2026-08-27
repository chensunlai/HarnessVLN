from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.config import load_symbol
from harness.errors import HarnessError, ToolClosedError
from harness.output import ModuleOutput, NULL_MODULE_OUTPUT
from harness.tool_bus import Tool
from schemas import (
    EnvironmentEpisode,
    EnvironmentTerminal,
    MotionProfile,
    NavigationProfile,
    Observation,
    Pose,
)


NativeFactory = Callable[[EnvironmentEpisode], Any]


class _GymRegistryCompat(dict[Any, Any]):
    """Expose the pre-0.26 registry attribute expected by Habitat-Lab 0.3.3."""

    @property
    def env_specs(self) -> "_GymRegistryCompat":
        return self


class HabitatEnvironment:
    def __init__(
        self,
        episode: EnvironmentEpisode,
        *,
        native_factory: NativeFactory,
        native_actions: Mapping[str, Any] | None = None,
        goal_finish_action: Any = 0,
        observation_channels: Sequence[str] = (
            "rgb",
            "depth",
            "gps",
            "compass",
        ),
        static_channels: Mapping[str, Any] | None = None,
        expose_pose: bool = True,
        forward_m: float = 0.25,
        turn_deg: float = 15.0,
        camera: Mapping[str, Any] | None = None,
        oracle_success_distance: float | None = None,
    ) -> None:
        self.episode = episode
        self.native_factory = native_factory
        self.native_actions = dict(
            native_actions
            or {"forward": 1, "turn_left": 2, "turn_right": 3, "look_up": 4, "look_down": 5}
        )
        self.goal_finish_action = goal_finish_action
        if oracle_success_distance is not None and oracle_success_distance <= 0:
            raise ValueError("oracle_success_distance must be positive")
        self.oracle_success_distance = oracle_success_distance
        self.observation_channels = tuple(observation_channels)
        self.static_channels = dict(static_channels or {})
        self.expose_pose = expose_pose
        provided_channels = set(self.observation_channels) | set(self.static_channels)
        if expose_pose:
            provided_channels.add("pose")
        self.profile = NavigationProfile(
            observation_channels=frozenset(provided_channels),
            motion=MotionProfile(
                "nav.move.discrete",
                frozenset(self.native_actions),
                frame="habitat_episode",
                units="meters_degrees",
                forward_m=forward_m,
                turn_deg=turn_deg,
            ),
            camera=dict(camera or {}),
        )
        self._forward_m = float(forward_m)
        self._turn_deg = float(turn_deg)
        self._session: Any = None
        self._observation: Mapping[str, Any] = {}
        self._running = False
        self._generation = 0
        self._lock = asyncio.Lock()
        self._terminal: asyncio.Future[EnvironmentTerminal] | None = None
        self._goal_stream = tuple(
            episode.setup.get("goal_stream", (episode.task.goal,))
        )
        self._goal_index = 0
        self._actions_this_goal = 0
        self._action_count = 0
        self._observation_id = 0
        self._metrics: dict[str, Any] = {}
        self._minimum_distance_to_goal: float | None = None
        self._final_pose: Pose | None = None
        self._output = NULL_MODULE_OUTPUT

    async def start(
        self, task, output: ModuleOutput = NULL_MODULE_OUTPUT
    ) -> Sequence[Tool]:
        del task
        if self._session is not None:
            raise HarnessError("Habitat environment instances are single-use")
        self._output = output
        self._session = self.native_factory(self.episode)
        self._observation = self._session.reset()
        self._running = True
        self._terminal = asyncio.get_running_loop().create_future()
        self._capture_metrics()
        output.record({"profile": self.profile.as_dict()})
        self._record_main_camera("reset")
        return (
            Tool(
                "nav.observe",
                "Get the current normalized Habitat navigation observation.",
                {"type": "object", "additionalProperties": False},
                self._observe,
            ),
            Tool(
                "nav.move.discrete",
                "Execute one Habitat discrete action.",
                {
                    "type": "object",
                    "properties": {"action": {"enum": sorted(self.native_actions)}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                self._move,
                writes=True,
            ),
            Tool(
                "nav.goal.finish",
                "Finish the current Habitat navigation goal.",
                {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["status"],
                    "additionalProperties": False,
                },
                self._finish_goal,
                writes=True,
            ),
        )

    async def _observe(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor, arguments
        self._ensure_running()
        self._observation_id += 1
        now = time.time()
        pose = self._pose()
        channels = {
            name: self._observation[name]
            for name in self.observation_channels
            if name in self._observation
        }
        channels.update(self.static_channels)
        if pose is not None:
            channels.setdefault("pose", pose.as_dict())
        return Observation(
            str(self._observation_id),
            now,
            now,
            "habitat_episode",
            channels,
            pose,
            {"goal_id": self._goal_stream[self._goal_index].goal_id},
        ).as_dict()

    async def _move(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        generation = self._generation
        async with self._lock:
            self._ensure_running()
            if generation != self._generation:
                raise ToolClosedError("stale Habitat motion generation")
            native_action = self.native_actions[arguments["action"]]
            pose_before = self._pose()
            if native_action is not None:
                self._observation = self._session.step(native_action)
            pose_after = self._pose()
            self._action_count += 1
            self._actions_this_goal += 1
            self._record_main_camera("action")
            self._capture_metrics()
            self._capture_native_terminal()
            return {
                "action": arguments["action"],
                "action_count": self._action_count,
                "goal_action_count": self._actions_this_goal,
                "native_terminal": self._native_terminal(),
                "motion": _motion_feedback(
                    arguments["action"],
                    pose_before,
                    pose_after,
                    forward_m=self._forward_m,
                    turn_deg=self._turn_deg,
                ),
            }

    async def _finish_goal(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        del actor
        async with self._lock:
            self._ensure_running()
            if arguments["status"] != "completed":
                return {"done": True, "accepted": False}
            self._observation = self._session.step(self.goal_finish_action)
            self._record_main_camera("goal_finish")
            self._capture_metrics()
            if self._goal_index + 1 >= len(self._goal_stream):
                return {"done": True, "accepted": True}
            self._goal_index += 1
            self._actions_this_goal = 0
            goal = self._goal_stream[self._goal_index]
            return {
                "done": False,
                "accepted": True,
                "goal": {
                    "goal_id": goal.goal_id,
                    "instruction": goal.instruction,
                    "modality": goal.modality,
                    "public": dict(goal.public),
                },
            }

    async def wait_terminal(self) -> EnvironmentTerminal:
        if self._terminal is None:
            raise HarnessError("Habitat environment is not started")
        return await self._terminal

    async def stop(self, reason: str) -> None:
        del reason
        async with self._lock:
            self._generation += 1
            self._running = False
            self._capture_metrics()
            self._final_pose = self._pose()
            if self._session is not None:
                self._session.close()

    def result(self) -> dict[str, Any]:
        value = {
            **self._metrics,
            "action_count": self._action_count,
            "goal_index": self._goal_index,
            "goal_count": len(self._goal_stream),
            "stopped": not self._running,
        }
        if self._final_pose is not None:
            value["final_pose"] = self._final_pose.as_dict()
        return value

    def _pose(self) -> Pose | None:
        if not self.expose_pose:
            return None
        if "gps" in self._observation:
            gps = self._observation["gps"]
            compass: Any = self._observation.get("compass", 0.0)
            try:
                yaw = float(compass[0])
            except (IndexError, TypeError):
                yaw = float(compass)
            return Pose(
                "habitat_episode",
                float(gps[0]),
                float(gps[1]),
                yaw=yaw,
            )
        sim = getattr(self._session, "sim", None)
        if sim is not None and hasattr(sim, "get_agent_state"):
            state = sim.get_agent_state()
            position = state.position
            return Pose(
                "habitat_world",
                float(position[0]),
                float(position[2]),
                float(position[1]),
            )
        return None

    def _capture_metrics(self) -> None:
        if self._session is not None and hasattr(self._session, "get_metrics"):
            value = self._session.get_metrics()
            if isinstance(value, Mapping):
                self._metrics = dict(value)
                distance = value.get("distance_to_goal")
                if isinstance(distance, (int, float)):
                    numeric_distance = float(distance)
                    if self._minimum_distance_to_goal is None:
                        self._minimum_distance_to_goal = numeric_distance
                    else:
                        self._minimum_distance_to_goal = min(
                            self._minimum_distance_to_goal, numeric_distance
                        )
                if (
                    self.oracle_success_distance is not None
                    and "oracle_success" not in self._metrics
                    and self._minimum_distance_to_goal is not None
                ):
                    self._metrics["oracle_success"] = float(
                        self._minimum_distance_to_goal
                        <= self.oracle_success_distance
                    )

    def _record_main_camera(self, stage: str) -> None:
        if "rgb" not in self.observation_channels or "rgb" not in self._observation:
            self._output.unavailable(
                "main_camera", "Habitat observation has no configured rgb channel"
            )
            return
        self._output.frame(
            "main_camera",
            self._observation["rgb"],
            {
                "source_time": time.time(),
                "stage": stage,
                "action_index": self._action_count,
                "goal_index": self._goal_index,
            },
        )

    def _native_terminal(self) -> bool:
        return bool(getattr(self._session, "episode_over", False))

    def _capture_native_terminal(self) -> None:
        if self._native_terminal() and self._terminal is not None and not self._terminal.done():
            self._terminal.set_result(
                EnvironmentTerminal("completed", "Habitat episode ended")
            )

    def _ensure_running(self) -> None:
        if not self._running:
            raise ToolClosedError("Habitat environment is stopped")


def from_episode(
    episode: EnvironmentEpisode,
    *,
    native_factory: str,
    native_factory_params: Mapping[str, Any] | None = None,
    **adapter_params: Any,
) -> HabitatEnvironment:
    factory = load_symbol(native_factory)

    def build(private_episode: EnvironmentEpisode) -> Any:
        return factory(private_episode, **dict(native_factory_params or {}))

    return HabitatEnvironment(episode, native_factory=build, **adapter_params)


def create_native_session(
    runtime_episode: EnvironmentEpisode,
    *,
    config_path: str | Path,
    config_loader: str = "envs.habitat:load_habitat_config",
    config_loader_params: Mapping[str, Any] | None = None,
    config_options: Sequence[Any] | None = None,
    config_values: Mapping[str, Any] | None = None,
    source_root: str | Path | None = None,
    scene_id_rewrites: Mapping[str, str] | None = None,
    scene_dataset_config: str | Path | None = None,
) -> Any:
    """Create a one-episode Habitat session inside a Habitat-enabled process."""

    if source_root is not None:
        _prepend_habitat_sources(source_root)
    ensure_habitat_gym_compat()
    try:
        import habitat
    except ImportError as error:
        raise HarnessError("Habitat adapter requires habitat-lab and habitat-sim") from error
    _quiet_habitat_loggers(habitat)
    loader = load_symbol(config_loader)
    config = loader(
        str(config_path),
        list(config_options or ()),
        **dict(config_loader_params or {}),
    )
    if config_values:
        _update_habitat_config(config, config_values)
    task_config = getattr(config, "TASK_CONFIG", None)
    if task_config is not None:
        dataset_config = task_config.DATASET
        dataset = habitat.make_dataset(dataset_config.TYPE, config=dataset_config)
        env_config = task_config
    elif hasattr(config, "habitat"):
        dataset_config = config.habitat.dataset
        dataset = habitat.make_dataset(dataset_config.type, config=dataset_config)
        env_config = config
    else:
        raise HarnessError("unsupported Habitat config shape")
    episode_id = runtime_episode.task.task_id.rsplit(":", 1)[-1]
    matches = [
        native_episode
        for native_episode in dataset.episodes
        if str(native_episode.episode_id) == episode_id
        and (
            runtime_episode.task.scene_id is None
            or _same_scene(
                str(native_episode.scene_id), runtime_episode.task.scene_id
            )
        )
    ]
    if len(matches) != 1:
        raise HarnessError(
            "expected one Habitat episode for "
            f"{runtime_episode.task.task_id}, found {len(matches)}"
        )
    native_episode = matches[0]
    if scene_id_rewrites:
        native_episode.scene_id = _rewrite_prefix(
            str(native_episode.scene_id), scene_id_rewrites
        )
    if scene_dataset_config is not None and hasattr(
        native_episode, "scene_dataset_config"
    ):
        native_episode.scene_dataset_config = str(scene_dataset_config)
    dataset.episodes = matches
    return habitat.Env(config=env_config, dataset=dataset)


def load_habitat_config(config_path: str, config_options: Sequence[Any]) -> Any:
    """Load a Habitat-Lab 0.3.x Hydra config without leaking Hydra into callers."""

    import habitat

    _quiet_habitat_loggers(habitat)
    return habitat.get_config(
        config_path=config_path,
        overrides=[str(option) for option in config_options],
    )


def _quiet_habitat_loggers(habitat: Any) -> None:
    habitat.logger.setLevel(logging.ERROR)
    try:
        import habitat_sim
    except ImportError:
        return
    habitat_sim.logging.logger.setLevel(logging.ERROR)


def ensure_habitat_gym_compat() -> None:
    """Bridge Habitat-Lab 0.3.3's registry lookup to Gym 0.26."""

    try:
        import gym
        from gym.envs import registration
    except ImportError as error:
        raise HarnessError("Habitat adapter requires gym") from error
    registry = registration.registry
    if isinstance(registry, dict) and not hasattr(registry, "env_specs"):
        compatible = _GymRegistryCompat(registry)
        registration.registry = compatible
        gym.envs.registry = compatible


def _prepend_habitat_sources(source_root: str | Path) -> None:
    root = Path(source_root).expanduser().resolve()
    habitat_lab = root / "habitat-lab" if (root / "habitat-lab").is_dir() else root
    if not (habitat_lab / "habitat").is_dir():
        raise HarnessError(f"Habitat-Lab source package not found under: {root}")
    candidates = [habitat_lab]
    baselines = root / "habitat-baselines"
    if baselines.is_dir():
        candidates.append(baselines)
    for candidate in reversed(candidates):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


def _update_habitat_config(config: Any, values: Mapping[str, Any]) -> None:
    try:
        from habitat.config import read_write
        from omegaconf import OmegaConf, open_dict
    except ImportError as error:
        raise HarnessError("Habitat config overrides require OmegaConf") from error
    with read_write(config), open_dict(config):
        for path, value in values.items():
            OmegaConf.update(config, path, value, merge=False)


def _same_scene(actual: str, expected: str) -> bool:
    actual_path = actual.replace("\\", "/").replace("//", "/").rstrip("/")
    expected_path = expected.replace("\\", "/").replace("//", "/").rstrip("/")
    return (
        actual_path == expected_path
        or actual_path.endswith(f"/{expected_path.lstrip('/')}")
        or expected_path.endswith(f"/{actual_path.lstrip('/')}")
    )


def _rewrite_prefix(value: str, rewrites: Mapping[str, str]) -> str:
    for source, destination in rewrites.items():
        if value.startswith(source):
            return f"{destination}{value[len(source):]}"
    return value


def _motion_feedback(
    action: str,
    before: Pose | None,
    after: Pose | None,
    *,
    forward_m: float,
    turn_deg: float,
) -> dict[str, Any] | None:
    if before is None or after is None or before.frame != after.frame:
        return None
    translation = math.dist(
        (before.x, before.y, before.z),
        (after.x, after.y, after.z),
    )
    rotation = (
        math.degrees(
            math.atan2(
                math.sin(after.yaw - before.yaw),
                math.cos(after.yaw - before.yaw),
            )
        )
        if before.yaw is not None and after.yaw is not None
        else None
    )
    blocked = False
    if action == "forward":
        blocked = translation < forward_m * 0.5
    elif action in {"turn_left", "turn_right"} and rotation is not None:
        blocked = abs(rotation) < turn_deg * 0.5
    return {
        "translation_m": round(translation, 4),
        "rotation_deg": round(rotation, 3) if rotation is not None else None,
        "blocked": blocked,
        "pose": after.as_dict(),
    }

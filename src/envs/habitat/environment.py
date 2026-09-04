from __future__ import annotations

import contextlib
import importlib
import logging
import math
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Mapping, Sequence

from domain.contracts import NavigationEpisode
from domain.errors import HarnessError
from envs.session import SessionEnvironment, _load


_NATIVE_OUTPUT_LOCK = threading.RLock()


class _GymRegistryCompat(dict[Any, Any]):
    @property
    def env_specs(self) -> "_GymRegistryCompat":
        return self


class HabitatEnvironment(SessionEnvironment):
    def __init__(
        self,
        *,
        config_path: str | Path,
        config_loader: str | None = None,
        config_overrides: Sequence[str] = (),
        source_root: str | Path | None = None,
        source_roots: Sequence[str | Path] = (),
        bootstrap: Sequence[str] = (),
        scene_id_rewrites: Mapping[str, str] | None = None,
        native_actions: Mapping[str, Any] | None = None,
        goal_action: Any | None = None,
        terminal_action: Any = 0,
        observation_channels: Sequence[str] = ("rgb", "depth", "gps", "compass"),
        main_camera_channel: str = "rgb",
        max_steps: int = 500,
        quiet_native_logs: bool = True,
    ) -> None:
        self.quiet_native_logs = quiet_native_logs
        super().__init__(
            session_factory="envs.habitat.environment:create_session",
            session_params={
                "config_path": str(config_path),
                "config_loader": config_loader,
                "config_overrides": list(config_overrides),
                "source_root": str(source_root) if source_root else None,
                "source_roots": [str(value) for value in source_roots],
                "bootstrap": list(bootstrap),
                "scene_id_rewrites": dict(scene_id_rewrites or {}),
                "quiet_native_logs": quiet_native_logs,
            },
            native_actions=native_actions
            or {"forward": 1, "turn_left": 2, "turn_right": 3, "look_up": 4, "look_down": 5},
            goal_action=goal_action,
            terminal_action=terminal_action,
            observation_channels=observation_channels,
            main_camera_channel=main_camera_channel,
            max_steps=max_steps,
        )

    def close(self) -> None:
        with _quiet_native_output(self.quiet_native_logs):
            super().close()

    def _pose(self) -> Any:
        gps = self._observation.get("gps")
        compass = self._observation.get("compass")
        if gps is None:
            return super()._pose()
        position: Any = None
        sim = getattr(self._session, "sim", None)
        if sim is not None and callable(getattr(sim, "get_agent_state", None)):
            native = sim.get_agent_state().position
            position = [float(native[0]), float(native[1]), float(native[2])]
        if position is None:
            position = [float(gps[0]), 0.0, float(gps[1])]
        try:
            heading = float(compass[0]) if compass is not None else 0.0
        except (IndexError, TypeError):
            heading = float(compass or 0.0)
        degrees = math.degrees(heading) % 360.0
        return {
            "frame": "habitat_world",
            "position": position,
            "gps": [float(gps[0]), float(gps[1])],
            "heading_degrees": (
                0.0 if math.isclose(degrees, 360.0, abs_tol=0.01) else round(degrees, 2)
            ),
        }


def create_session(
    episode: NavigationEpisode,
    *,
    config_path: str,
    config_loader: str | None = None,
    config_overrides: Sequence[str] = (),
    source_root: str | None = None,
    source_roots: Sequence[str] = (),
    bootstrap: Sequence[str] = (),
    scene_id_rewrites: Mapping[str, str] | None = None,
    quiet_native_logs: bool = True,
) -> Any:
    for source_value in ([source_root] if source_root else []) + list(source_roots):
        source = Path(source_value).expanduser().resolve()
        candidates = (source, source / "habitat-lab", source / "habitat-baselines")
        for candidate in reversed(candidates):
            value = str(candidate)
            if candidate.is_dir() and value not in sys.path:
                sys.path.insert(0, value)
    with _quiet_native_output(quiet_native_logs):
        _ensure_habitat_gym_compat()
        for module_name in bootstrap:
            importlib.import_module(module_name)
        try:
            import habitat
        except ImportError as error:
            raise HarnessError("Habitat environment requires habitat-lab and habitat-sim") from error
    _quiet_habitat_loggers(habitat)
    if config_loader:
        config = _load(config_loader)(config_path, list(config_overrides))
    else:
        config = _default_config(habitat, config_path, config_overrides)
    habitat_config = getattr(config, "habitat", config)
    dataset_config = getattr(habitat_config, "dataset", getattr(config, "DATASET", None))
    if dataset_config is None:
        raise HarnessError("Habitat config has no dataset section")
    dataset_type = getattr(dataset_config, "type", getattr(dataset_config, "TYPE", None))
    dataset = habitat.make_dataset(dataset_type, config=dataset_config)
    source_id = str(episode.setup.get("source_episode_id", episode.episode_id))
    rewrites = dict(scene_id_rewrites or {})
    selected = []
    for native in dataset.episodes:
        scene_id = str(getattr(native, "scene_id", ""))
        for old, new in rewrites.items():
            if scene_id.startswith(old):
                native.scene_id = new + scene_id[len(old) :]
                scene_id = native.scene_id
        expected_scene = str(episode.scene_id or episode.setup.get("scene_id", ""))
        scene_matches = not expected_scene or (
            Path(scene_id).name == Path(expected_scene).name
            or scene_id.endswith(expected_scene)
            or expected_scene.endswith(scene_id)
        )
        if str(getattr(native, "episode_id", "")) == source_id and scene_matches:
            selected.append(native)
    if not selected and "native_episode_index" in episode.setup:
        index = int(episode.setup["native_episode_index"])
        if 0 <= index < len(dataset.episodes):
            selected = [dataset.episodes[index]]
    if not selected:
        raise HarnessError(f"Habitat dataset has no episode {source_id!r}")
    dataset.episodes = selected[:1]
    with _quiet_native_output(quiet_native_logs):
        try:
            return habitat.Env(config=habitat_config, dataset=dataset)
        except TypeError:
            return habitat.Env(config=config, dataset=dataset)


def _default_config(habitat: Any, path: str, overrides: Sequence[str]) -> Any:
    if callable(getattr(habitat, "get_config", None)):
        try:
            return habitat.get_config(path, overrides=list(overrides))
        except TypeError:
            return habitat.get_config(config_paths=path, overrides=list(overrides))
    try:
        from habitat.config.default import get_config

        return get_config(path, list(overrides))
    except ImportError as error:
        raise HarnessError("installed Habitat version exposes no config loader") from error


def _ensure_habitat_gym_compat() -> None:
    try:
        import gym
        from gym.envs import registration
    except ImportError as error:
        raise HarnessError("Habitat environment requires gym") from error
    registry = registration.registry
    if isinstance(registry, dict) and not hasattr(registry, "env_specs"):
        compatible = _GymRegistryCompat(registry)
        registration.registry = compatible
        gym.envs.registry = compatible


def _quiet_habitat_loggers(habitat: Any) -> None:
    habitat.logger.setLevel(logging.ERROR)
    try:
        import habitat_sim
    except ImportError:
        return
    habitat_sim.logging.logger.setLevel(logging.ERROR)


@contextlib.contextmanager
def _quiet_native_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with _NATIVE_OUTPUT_LOCK:
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        null_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fds = (os.dup(1), os.dup(2))
        try:
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)
            yield
        finally:
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            os.close(saved_fds[0])
            os.close(saved_fds[1])
            os.close(null_fd)

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from domain.contracts import NavigationEpisode
from domain.errors import HarnessError
from envs.session import SessionEnvironment, _resolve


class IsaacEnvironment(SessionEnvironment):
    """Adapter for VLN-PE/VLNVerse vector sessions backed by Isaac Sim."""

    def __init__(
        self,
        *,
        runtime_factory: str,
        runtime_params: Mapping[str, Any] | None = None,
        native_actions: Mapping[str, Any] | None = None,
        warmup_action: Mapping[str, Any] | None = None,
        terminal_action: Any | None = None,
        flash: bool = False,
        observation_channels: Sequence[str] = ("rgb", "depth", "pose", "metrics"),
        main_camera_channel: str = "rgb",
        max_steps: int = 500,
        max_native_ticks_per_action: int = 2000,
    ) -> None:
        if max_native_ticks_per_action < 1:
            raise ValueError("max_native_ticks_per_action must be positive")
        controller = "move_by_flash" if flash else "move_by_discrete"
        super().__init__(
            session_factory="envs.isaac.environment:create_session",
            session_params={
                "runtime_factory": runtime_factory,
                "runtime_params": dict(runtime_params or {}),
            },
            native_actions=native_actions
            or {
                "stand_still": {"h1": {"stand_still": []}},
                "forward": {"h1": {controller: [1]}},
                "turn_left": {"h1": {controller: [2]}},
                "turn_right": {"h1": {controller: [3]}},
            },
            goal_action=None,
            terminal_action=terminal_action or {"h1": {"stop": []}},
            observation_channels=observation_channels,
            main_camera_channel=main_camera_channel,
            max_steps=max_steps,
        )
        self.warmup_action = dict(warmup_action or {"h1": {"stand_still": []}})
        self.max_native_ticks_per_action = max_native_ticks_per_action
        self._native_tick_count = 0

    def start(self) -> None:
        super().start()
        observation, terminal, info = self._native_step(self.warmup_action)
        if terminal:
            raise HarnessError("Isaac environment terminated during warmup")
        self._observation = observation
        self._capture_metrics(info)

    def _native_step(
        self, native_action: Any
    ) -> tuple[Mapping[str, Any], bool, Mapping[str, Any]]:
        if self._session is None or not callable(getattr(self._session, "step", None)):
            raise HarnessError("Isaac runtime session must implement step()")
        for _ in range(self.max_native_ticks_per_action):
            value = _resolve(self._session.step([native_action]))
            observation, terminal, info = self._normalize_step(value)
            self._native_tick_count += 1
            if terminal or bool(observation.get("finish_action", False)):
                return observation, terminal, info
        raise HarnessError(
            "Isaac action did not finish within max_native_ticks_per_action"
        )

    def result(self) -> Mapping[str, Any]:
        value = dict(super().result())
        value["native_tick_count"] = self._native_tick_count
        value["native_metrics"] = dict(self._metrics)
        value.update(_metric_scalars(self._metrics))
        return value

    def _normalize_step(
        self, value: Any, *, reset: bool = False
    ) -> tuple[Mapping[str, Any], bool, Mapping[str, Any]]:
        terminal = False
        info: Mapping[str, Any] = {}
        observation = value
        if isinstance(value, tuple):
            observation = value[0]
            if not reset and len(value) >= 3:
                terminal = _first_bool(value[2])
                if len(value) >= 4:
                    terminal = terminal or _first_bool(value[3])
                if len(value) >= 5 and isinstance(value[4], Mapping):
                    info = value[4]
            elif len(value) >= 2 and isinstance(value[1], Mapping):
                info = value[1]
        observation = _first(observation)
        if isinstance(observation, Mapping) and "h1" in observation:
            observation = observation["h1"]
        if not isinstance(observation, Mapping):
            raise HarnessError("Isaac observation must resolve to one robot mapping")
        return observation, terminal, info


def create_session(
    episode: NavigationEpisode,
    *,
    runtime_factory: str,
    runtime_params: Mapping[str, Any],
) -> Any:
    module_name, separator, attribute = runtime_factory.partition(":")
    if not separator:
        raise HarnessError(f"invalid Isaac runtime factory: {runtime_factory}")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        return factory(episode=episode, **dict(runtime_params))
    except (ImportError, AttributeError, TypeError) as error:
        raise HarnessError(f"failed to create Isaac runtime: {error}") from error


def _first(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        if len(value) != 1:
            raise HarnessError("Isaac adapter expects a vector environment of size one")
        return value[0]
    return value


def _first_bool(value: Any) -> bool:
    value = _first(value)
    try:
        return bool(value.item())
    except AttributeError:
        return bool(value)


def _metric_scalars(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aliases = {"success": "success", "spl": "spl", "NE": "NE", "osr": "osr", "ndtw": "ndtw"}
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                if key in aliases and isinstance(item, (int, float, bool)):
                    result[aliases[key]] = item
                else:
                    stack.append(item)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            stack.extend(current)
    return result

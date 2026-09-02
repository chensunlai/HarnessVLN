from __future__ import annotations

import asyncio
import importlib
import inspect
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from domain.errors import HarnessError
from domain.modules import EnvironmentModule


class SessionEnvironment(EnvironmentModule):
    """Normalize a reset/step/close Python session into the Domain Env contract."""

    def __init__(
        self,
        *,
        session_factory: str,
        session_params: Mapping[str, Any] | None = None,
        native_actions: Mapping[str, Any] | None = None,
        goal_action: Any | None = None,
        terminal_action: Any | None = None,
        observation_channels: Sequence[str] = (),
        main_camera_channel: str | None = "rgb",
        max_steps: int = 500,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.session_factory = session_factory
        self.session_params = dict(session_params or {})
        self.native_actions = dict(native_actions or {})
        self.goal_action = goal_action
        self.terminal_action = terminal_action
        self.observation_channels = tuple(observation_channels)
        self.main_camera_channel = main_camera_channel
        self.max_steps = max_steps
        self._session: Any = None
        self._observation: Mapping[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._actions: list[str] = []
        self._observation_id = 0
        self._goal_index = 0
        self._lock = threading.RLock()
        self._stopped = False

    def mount(self) -> None:
        register = self.context.register
        self.expose(
            "env.observe",
            self.observe,
            description="Read the current normalized navigation observation.",
            parameters={"type": "object", "additionalProperties": False},
        )
        action_schema: dict[str, Any] = {"type": "string"}
        if self.native_actions:
            action_schema["enum"] = sorted(self.native_actions)
        self.expose(
            "env.step",
            self.step,
            description="Execute one normalized navigation action.",
            parameters={
                "type": "object",
                "properties": {"action": action_schema},
                "required": ["action"],
                "additionalProperties": False,
            },
            mutates=True,
            serial_key="env.state",
        )
        if self.goal_action is not None:
            self.expose(
                "env.goal.finish",
                self.finish_goal,
                description="Finish the current goal without resetting the Domain.",
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["status"],
                    "additionalProperties": False,
                },
                mutates=True,
                serial_key="env.state",
            )
        register.register_reference(
            self.context.name,
            "env.observation",
            lambda: self._observation,
            description="Latest native observation mapping.",
        )
        register.register_reference(
            self.context.name,
            "env.pose",
            self._pose,
            description="Latest pose if exposed by the backend.",
        )
        register.register_reference(
            self.context.name,
            "env.main_camera",
            self._main_camera,
            description="Latest frame from the environment-selected main camera.",
        )
        register.register_reference(
            self.context.name,
            "env.capabilities",
            lambda: {
                "actions": sorted(self.native_actions),
                "observation_channels": list(self.observation_channels),
                "main_camera_channel": self.main_camera_channel,
                "multi_goal": self.goal_action is not None,
            },
            description="Normalized environment capabilities.",
        )

    def start(self) -> None:
        factory = _load(self.session_factory)
        self._session = _resolve(factory(self.context.episode, **self.session_params))
        if not callable(getattr(self._session, "reset", None)):
            raise HarnessError("native environment session must implement reset()")
        self._observation, _, info = self._normalize_step(
            _resolve(self._session.reset()), reset=True
        )
        self._capture_metrics(info)
        self.context.metadata.update(
            {
                "session_factory": self.session_factory,
                "actions": sorted(self.native_actions),
                "main_camera_channel": self.main_camera_channel,
            }
        )

    def observe(self) -> dict[str, Any]:
        self._ensure_ready()
        with self._lock:
            self._observation_id += 1
            channels = (
                {
                    name: self._observation[name]
                    for name in self.observation_channels
                    if name in self._observation
                }
                if self.observation_channels
                else dict(self._observation)
            )
            return {
                "observation_id": self._observation_id,
                "channels": channels,
                "pose": self._pose(),
                "action_count": len(self._actions),
            }

    def step(self, action: str) -> dict[str, Any]:
        self._ensure_ready()
        if action not in self.native_actions:
            raise HarnessError(f"unsupported environment action: {action!r}")
        with self._lock:
            if len(self._actions) >= self.max_steps:
                self.finish("failed", "maximum action count reached", "env")
                return {"accepted": False, "terminal": True}
            observation, terminal, info = self._native_step(self.native_actions[action])
            self._observation = observation
            self._actions.append(action)
            self._capture_metrics(info)
            if terminal:
                self.finish("environment_terminal", "native environment ended", "env")
            return {
                "accepted": True,
                "action": action,
                "action_count": len(self._actions),
                "terminal": terminal,
                "pose": self._pose(),
            }

    def stop(self, reason: str) -> None:
        del reason
        with self._lock:
            if self._stopped:
                return
            if self._session is not None and self.terminal_action is not None:
                observation, _, info = self._native_step(self.terminal_action)
                self._observation = observation
                self._capture_metrics(info)
            native_stop = getattr(self._session, "stop", None)
            if callable(native_stop):
                _resolve(native_stop())
            self._stopped = True

    def finish_goal(self, status: str, reason: str = "") -> dict[str, Any]:
        del reason
        self._ensure_ready()
        if status != "completed":
            return {"accepted": False, "done": False, "goal_index": self._goal_index}
        with self._lock:
            observation, terminal, info = self._native_step(self.goal_action)
            self._observation = observation
            self._capture_metrics(info)
            self._goal_index += 1
            goals = self.context.episode.public.get("goals", ())
            next_goal = (
                goals[self._goal_index]
                if isinstance(goals, Sequence) and self._goal_index < len(goals)
                else None
            )
            if terminal:
                self.finish("environment_terminal", "native environment ended", "env")
            return {
                "accepted": True,
                "done": terminal,
                "goal_index": self._goal_index,
                "goal": next_goal,
            }

    def close(self) -> None:
        with self._lock:
            session, self._session = self._session, None
            if session is not None and callable(getattr(session, "close", None)):
                _resolve(session.close())
            self._stopped = True

    def result(self) -> Mapping[str, Any]:
        return {
            **self._metrics,
            "action_count": len(self._actions),
            "actions": list(self._actions),
            "goal_index": self._goal_index,
            "final_pose": self._pose(),
            "stopped": self._stopped,
        }

    def _native_step(self, native_action: Any) -> tuple[Mapping[str, Any], bool, Mapping[str, Any]]:
        if self._session is None or not callable(getattr(self._session, "step", None)):
            raise HarnessError("native environment session must implement step()")
        value = _resolve(self._session.step(native_action))
        observation, terminal, info = self._normalize_step(value)
        terminal = terminal or bool(getattr(self._session, "episode_over", False))
        return observation, terminal, info

    def _normalize_step(
        self, value: Any, *, reset: bool = False
    ) -> tuple[Mapping[str, Any], bool, Mapping[str, Any]]:
        del reset
        terminal = False
        info: Mapping[str, Any] = {}
        observation = value
        if isinstance(value, tuple):
            if len(value) == 5:
                observation, _, terminated, truncated, info = value
                terminal = bool(terminated) or bool(truncated)
            elif len(value) == 4:
                observation, _, terminal, info = value
            elif len(value) == 2:
                observation, info = value
            elif len(value) == 1:
                observation = value[0]
        if not isinstance(observation, Mapping):
            raise HarnessError(
                f"native observation must be a mapping, got {type(observation).__name__}"
            )
        return observation, bool(terminal), info if isinstance(info, Mapping) else {}

    def _capture_metrics(self, info: Mapping[str, Any]) -> None:
        native = getattr(self._session, "get_metrics", None)
        if callable(native):
            value = native()
            if isinstance(value, Mapping):
                self._metrics.update(value)
        metrics = info.get("metrics") if isinstance(info, Mapping) else None
        if isinstance(metrics, Mapping):
            self._metrics.update(metrics)
        observation_metrics = self._observation.get("metrics")
        if isinstance(observation_metrics, Mapping):
            self._metrics.update(observation_metrics)
        for name, value in info.items():
            if isinstance(name, str) and (
                isinstance(value, (str, int, float, bool)) or value is None
            ):
                self._metrics[name] = value

    def _pose(self) -> Any:
        for name in ("pose", "agent_pose", "gps"):
            if name in self._observation:
                return self._observation[name]
        return None

    def _main_camera(self) -> Any:
        if self.main_camera_channel is None:
            return None
        return self._observation.get(self.main_camera_channel)

    def _ensure_ready(self) -> None:
        if not self.wait_ready(0) or self._session is None:
            raise HarnessError("environment is not ready")
        if self.wait_terminal(0) is not None:
            raise HarnessError("environment is terminal")


def _load(target: str) -> Any:
    module, separator, attribute = target.partition(":")
    if not separator:
        raise HarnessError(f"invalid session factory {target!r}")
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise HarnessError(f"failed to load session factory {target!r}: {error}") from error


def _resolve(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(value))
    raise HarnessError("async native sessions must run in their dedicated module thread")


async def _await_value(value: Any) -> Any:
    return await value

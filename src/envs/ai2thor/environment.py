from __future__ import annotations

import importlib
import math
from typing import Any, Mapping

from domain.contracts import NavigationEpisode
from domain.errors import HarnessError
from envs.session import SessionEnvironment


class RoboTHOREnvironment(SessionEnvironment):
    def __init__(
        self,
        *,
        controller_factory: str = "ai2thor.controller:Controller",
        controller_params: Mapping[str, Any] | None = None,
        max_steps: int = 500,
        observation_channels: tuple[str, ...] = ("rgb", "depth", "pose"),
    ) -> None:
        super().__init__(
            session_factory="envs.ai2thor.environment:create_session",
            session_params={
                "controller_factory": controller_factory,
                "controller_params": dict(controller_params or {}),
            },
            native_actions={
                "forward": "MoveAhead",
                "backward": "MoveBack",
                "turn_left": "RotateLeft",
                "turn_right": "RotateRight",
                "look_up": "LookUp",
                "look_down": "LookDown",
            },
            terminal_action="Done",
            observation_channels=observation_channels,
            main_camera_channel="rgb",
            max_steps=max_steps,
        )


def create_session(
    episode: NavigationEpisode,
    *,
    controller_factory: str,
    controller_params: Mapping[str, Any],
) -> "RoboTHORSession":
    module_name, separator, attribute = controller_factory.partition(":")
    if not separator:
        raise HarnessError(f"invalid AI2-THOR controller factory: {controller_factory}")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        controller = factory(**dict(controller_params))
    except (ImportError, AttributeError, TypeError) as error:
        raise HarnessError(f"failed to create AI2-THOR controller: {error}") from error
    return RoboTHORSession(controller, episode)


class RoboTHORSession:
    def __init__(self, controller: Any, episode: NavigationEpisode) -> None:
        self.controller = controller
        self.episode = episode
        self._metrics: dict[str, Any] = {}
        self._last_position: tuple[float, float, float] | None = None
        self._path_length = 0.0
        self.episode_over = False

    def reset(self) -> Mapping[str, Any]:
        setup = self.episode.setup
        scene = str(setup.get("scene", self.episode.scene_id or ""))
        self.controller.reset(scene)
        position = setup.get("initial_position", {})
        orientation = setup.get("initial_orientation", 0)
        rotation = orientation if isinstance(orientation, Mapping) else {"y": orientation}
        arguments = {
            "action": "TeleportFull",
            "position": position,
            "rotation": rotation,
            "horizon": float(setup.get("initial_horizon", 0)),
            "standing": True,
        }
        event = self.controller.step(**arguments)
        return self._observation(event)

    def step(self, action: Any) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        if action == "Done":
            self.episode_over = True
            event = getattr(self.controller, "last_event", None)
            if event is None:
                event = self.controller.step(action="Pass")
            success = self._goal_visible(event)
            self._metrics.update(success=success, path_length=self._path_length)
            return self._observation(event), 0.0, True, dict(self._metrics)
        event = self.controller.step(action=action)
        observation = self._observation(event)
        self._update_path(observation.get("pose"))
        return observation, 0.0, False, {}

    def get_metrics(self) -> Mapping[str, Any]:
        return dict(self._metrics)

    def close(self) -> None:
        self.controller.stop()

    def _observation(self, event: Any) -> dict[str, Any]:
        metadata = getattr(event, "metadata", {})
        agent = metadata.get("agent", {}) if isinstance(metadata, Mapping) else {}
        pose = {
            "position": agent.get("position"),
            "rotation": agent.get("rotation"),
            "horizon": agent.get("cameraHorizon"),
        }
        value = {"rgb": getattr(event, "frame", None), "pose": pose, "metadata": metadata}
        depth = getattr(event, "depth_frame", None)
        if depth is not None:
            value["depth"] = depth
        return value

    def _goal_visible(self, event: Any) -> bool:
        target = self.episode.setup.get("object_type")
        metadata = getattr(event, "metadata", {})
        objects = metadata.get("objects", ()) if isinstance(metadata, Mapping) else ()
        return any(
            item.get("objectType") == target and bool(item.get("visible"))
            for item in objects
            if isinstance(item, Mapping)
        )

    def _update_path(self, pose: Any) -> None:
        if not isinstance(pose, Mapping) or not isinstance(pose.get("position"), Mapping):
            return
        value = pose["position"]
        point = (float(value.get("x", 0)), float(value.get("y", 0)), float(value.get("z", 0)))
        if self._last_position is not None:
            self._path_length += math.dist(self._last_position, point)
        self._last_position = point

from __future__ import annotations

import threading
from typing import Any


class AsyncSession:
    def __init__(self) -> None:
        self.owner_thread = threading.get_ident()
        self.position = 0
        self.done = False
        self.closed = False

    async def reset(self) -> dict[str, Any]:
        assert threading.get_ident() == self.owner_thread
        return {"rgb": [[0]], "pose": [self.position, 0, 0]}

    async def step(self, action: str):
        assert threading.get_ident() == self.owner_thread
        if action == "native_forward":
            self.position += 1
        if action == "native_stop":
            self.done = True
        observation = {"rgb": [[self.position]], "pose": [self.position, 0, 0]}
        return observation, 0.0, self.done, {"success": self.done}

    def get_metrics(self) -> dict[str, Any]:
        return {"success": self.done, "distance_to_goal": max(0, 2 - self.position)}

    async def close(self) -> None:
        assert threading.get_ident() == self.owner_thread
        self.closed = True


async def create(episode, **params):
    del episode, params
    return AsyncSession()


class MultiGoalSession:
    def __init__(self) -> None:
        self.goal_index = 0

    def reset(self):
        return {"rgb": [[0]], "pose": [0, 0, 0]}

    def step(self, action):
        if action == "finish_goal":
            self.goal_index += 1
        done = self.goal_index == 2
        return (
            {"rgb": [[self.goal_index]], "pose": [self.goal_index, 0, 0]},
            0.0,
            done,
            {"goals_completed": self.goal_index, "success": done},
        )

    def close(self):
        pass


def create_multi_goal(episode, **params):
    del episode, params
    return MultiGoalSession()


class IsaacSession:
    def __init__(self) -> None:
        self.tick = 0
        self.current_ticks = 0
        self.actions = 0
        self.stopping = False

    def reset(self):
        return [{"h1": {"rgb": [[0]], "finish_action": False}}]

    def step(self, actions):
        action = actions[0]["h1"]
        if self.current_ticks == 0:
            self.actions += 1
            self.stopping = "stop" in action
        self.tick += 1
        self.current_ticks += 1
        finished = self.current_ticks == 3
        terminal = finished and self.stopping
        observation = {
            "h1": {
                "rgb": [[self.tick]],
                "finish_action": finished,
                "metrics": {"success": terminal, "spl": 1.0 if terminal else 0.0},
            }
        }
        if finished:
            self.current_ticks = 0
        return [observation], [0.0], [terminal], [False], {}

    def close(self):
        pass


def create_isaac(episode, **params):
    del episode, params
    return IsaacSession()

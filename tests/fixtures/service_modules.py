from __future__ import annotations

import threading
import time

from domain.modules import Module
from envs.replay import ReplayEnvironment


class SlowStartReplayEnvironment(ReplayEnvironment):
    def __init__(self, *, start_delay_s: float) -> None:
        super().__init__()
        self.start_delay_s = start_delay_s

    def start(self) -> None:
        time.sleep(self.start_delay_s)
        super().start()


class EchoService(Module):
    def mount(self) -> None:
        self.expose(
            "echo.call",
            self.echo,
            description="Echo a value from the service module thread.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

    def echo(self, value: str):
        return {"value": value, "thread": threading.current_thread().name}


class FailingModule(Module):
    def run(self) -> None:
        raise RuntimeError("backend unavailable")


class ServiceDriver(Module):
    def run(self) -> None:
        result = self.context.register.call(
            self.context.name, "echo.call", {"value": "ready"}
        )
        self.context.output.write_json("echo.json", result)
        self.context.register.call(
            self.context.name,
            "env.stop",
            {
                "status": "completed",
                "reason": "service call completed",
                "actor": self.context.name,
            },
        )

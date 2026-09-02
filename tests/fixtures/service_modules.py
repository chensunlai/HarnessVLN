from __future__ import annotations

import threading

from domain.modules import Module


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


class ServiceDriver(Module):
    def run(self) -> None:
        result = self.context.register.call(
            self.context.name, "echo.call", {"value": "ready"}
        )
        self.context.output.write_json("echo.json", result)
        self.context.register.call(
            self.context.name,
            "env.stop",
            {"status": "completed", "reason": "service call completed", "actor": self.context.name},
        )

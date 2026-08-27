from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from harness.errors import HarnessError
from harness.output import ModuleOutput, NULL_MODULE_OUTPUT
from harness.tool_bus import Tool, ToolClient
from schemas import NavTask


@dataclass(slots=True)
class _Job:
    job_id: str
    instruction: str
    state: str = "running"
    steps: int = 0
    reason: str = ""
    task: asyncio.Task[None] | None = None


class DummyVLNNavigator:
    """A blocking VLN tool backed by an internally asynchronous policy loop."""

    required_tools = frozenset({"nav.observe", "nav.move.discrete"})
    requirements = {
        "observation_channels": ["target_delta", "pose"],
        "motion": {
            "tool": "nav.move.discrete",
            "actions": ["forward", "backward"],
            "frame": "dummy_world",
            "units": "meters_degrees",
            "forward_m": 1.0,
        },
    }

    def __init__(
        self,
        *,
        max_steps: int = 100,
        local_max_steps: int = 16,
        inference_period_s: float = 0.0,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if local_max_steps <= 0:
            raise ValueError("local_max_steps must be positive")
        self.max_steps = max_steps
        self.local_max_steps = local_max_steps
        self.inference_period_s = inference_period_s
        self._tools: ToolClient | None = None
        self._jobs: dict[str, _Job] = {}
        self._closed = False

    async def start(
        self,
        task: NavTask,
        tools: ToolClient,
        output: ModuleOutput = NULL_MODULE_OUTPUT,
    ):
        del task
        self._tools = tools
        output.record(
            {
                "navigator": type(self).__name__,
                "requirements": self.requirements,
                "local_max_steps": self.local_max_steps,
            }
        )
        return (
            Tool(
                "vln.navigate.task",
                "Run the complete task instruction with the VLN model and block "
                "until it finishes. This compatibility tool is intended for "
                "passthrough agents.",
                {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "minLength": 1},
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
                self._navigate_task,
                writes=True,
            ),
            Tool(
                "vln.navigate.local",
                "Navigate to one landmark, opening, object, or free-space region "
                "visible in the latest observation, then block until the local "
                "attempt finishes. The instruction must not contain the complete "
                "task or unseen waypoints.",
                {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                            "description": (
                                "One short instruction grounded in a target visible "
                                "in the latest observation."
                            ),
                        },
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
                self._navigate_local,
                writes=True,
            ),
        )

    async def _navigate_task(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._navigate(actor, arguments["instruction"], self.max_steps)

    async def _navigate_local(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._navigate(
            actor, arguments["instruction"], self.local_max_steps
        )

    async def _navigate(
        self, actor: str, instruction: str, max_steps: int
    ) -> dict[str, Any]:
        del actor
        if self._closed:
            raise HarnessError("navigator is stopped")
        if any(
            job.task is not None and not job.task.done()
            for job in self._jobs.values()
        ):
            raise HarnessError("navigator already has an active navigation job")
        job_id = uuid.uuid4().hex
        job = _Job(job_id, instruction)
        self._jobs[job_id] = job
        job.task = asyncio.create_task(
            self._run_job(job, max_steps), name=f"dummy-vln-{job_id}"
        )
        await job.task
        return {"state": job.state, "steps": job.steps, "reason": job.reason}

    async def _run_job(self, job: _Job, max_steps: int) -> None:
        assert self._tools is not None
        try:
            while job.steps < max_steps:
                observation = await self._tools.call("nav.observe")
                delta = int(observation["channels"]["target_delta"])
                if delta == 0:
                    job.state = "succeeded"
                    job.reason = "target reached"
                    return
                action = "forward" if delta > 0 else "backward"
                await self._tools.call("nav.move.discrete", action=action)
                job.steps += 1
                await asyncio.sleep(self.inference_period_s)
            job.state = "failed"
            job.reason = "maximum VLN steps reached"
        except asyncio.CancelledError:
            job.state = "cancelled"
            job.reason = "job cancelled"
            raise
        except Exception as error:
            job.state = "failed"
            job.reason = f"{type(error).__name__}: {error}"

    async def stop(self, reason: str) -> None:
        del reason
        self._closed = True
        tasks = []
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.state = "cancelled"
                job.reason = "navigator stopped"
                job.task.cancel()
                tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

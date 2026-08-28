from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from harness import Component, ComponentContext, Function


@dataclass(slots=True)
class _Job:
    state: str = "running"
    steps: int = 0
    error: str | None = None
    task: asyncio.Task[None] | None = None


class DummyVLN(Component):
    """A complete asynchronous navigator whose internal loop remains model-owned."""

    name = "vln"
    required_functions = frozenset({"nav.observe", "nav.move"})
    optional_functions = frozenset({"memory.remember"})

    def __init__(self, step_delay_s: float = 0.001) -> None:
        if step_delay_s < 0:
            raise ValueError("step_delay_s must not be negative")
        self.step_delay_s = step_delay_s
        self._jobs: dict[str, _Job] = {}
        self._context: ComponentContext | None = None

    def functions(self):
        job_input = {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
            "additionalProperties": False,
        }
        return (
            Function(
                "vln.navigate.start",
                "Start one complete VLN navigation Job.",
                self._start_job,
                input_schema={
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string"},
                        "options": {"type": "object"},
                    },
                    "required": ["instruction", "options"],
                    "additionalProperties": False,
                },
                mutates=True,
                serial_key="vln",
            ),
            Function(
                "vln.navigate.status",
                "Read a VLN Job state.",
                self._status,
                input_schema=job_input,
            ),
            Function(
                "vln.navigate.cancel",
                "Cancel a VLN Job and drain its navigation loop.",
                self._cancel,
                input_schema=job_input,
                mutates=True,
                serial_key="vln",
            ),
        )

    async def start(self, context: ComponentContext) -> None:
        self._context = context
        context.output.set_metadata(kind="asynchronous_navigation_job")

    async def _start_job(self, _call, arguments):
        if self._context is None:
            raise RuntimeError("VLN component is not started")
        if any(job.state == "running" for job in self._jobs.values()):
            raise RuntimeError("DummyVLN supports one active Job")
        job_id = uuid.uuid4().hex
        job = _Job()
        self._jobs[job_id] = job
        job.task = asyncio.create_task(
            self._navigate(job_id, job), name=f"dummy-vln:{job_id}"
        )
        return {"job_id": job_id}

    async def _status(self, _call, arguments):
        job = self._job(arguments["job_id"])
        return {"state": job.state, "steps": job.steps, "error": job.error}

    async def _cancel(self, _call, arguments):
        job = self._job(arguments["job_id"])
        if job.task is not None and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        if job.state == "running":
            job.state = "cancelled"
        return {"state": job.state}

    async def _navigate(self, job_id: str, job: _Job) -> None:
        assert self._context is not None
        target = int(self._context.task.metadata.get("target", 0))
        try:
            while not self._context.cancelled.is_set():
                observation = await self._context.functions.call("nav.observe")
                position = int(observation["position"])
                if position == target:
                    job.state = "completed"
                    return
                delta = 1 if position < target else -1
                moved = await self._context.functions.call("nav.move", delta=delta)
                job.steps += 1
                self._context.output.append_jsonl(
                    "inference/trace.jsonl",
                    {"job_id": job_id, "position": moved["position"], "delta": delta},
                )
                if self._context.functions.has("memory.remember"):
                    await self._context.functions.call(
                        "memory.remember",
                        text=f"visited position {moved['position']}",
                        position=moved["position"],
                    )
                await asyncio.sleep(self.step_delay_s)
            job.state = "cancelled"
        except asyncio.CancelledError:
            job.state = "cancelled"
            raise
        except Exception as error:
            job.state = "failed"
            job.error = f"{type(error).__name__}: {error}"

    def _job(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise ValueError(f"unknown VLN Job {job_id!r}") from error

    async def close(self, reason: str) -> None:
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.task.cancel()
        await asyncio.gather(
            *(job.task for job in self._jobs.values() if job.task is not None),
            return_exceptions=True,
        )
        if self._context is not None:
            trace = self._context.output.path("inference/trace.jsonl")
            if trace is not None and trace.exists():
                self._context.output.add_artifact(
                    "inference/trace.jsonl", "application/jsonl"
                )
            self._context.output.set_metadata(jobs=len(self._jobs))

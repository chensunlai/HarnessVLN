from __future__ import annotations

import asyncio

from harness import Agent, ComponentContext


class DummyAgent(Agent):
    """A minimal Agent driver that delegates navigation to one complete VLN Job."""

    name = "agent"
    required_functions = frozenset(
        {
            "nav.stop",
            "vln.navigate.start",
            "vln.navigate.status",
            "vln.navigate.cancel",
        }
    )

    def __init__(self, poll_interval_s: float = 0.001) -> None:
        if poll_interval_s < 0:
            raise ValueError("poll_interval_s must not be negative")
        self.poll_interval_s = poll_interval_s

    async def run(self, context: ComponentContext) -> None:
        context.output.set_metadata(driver="vln_job")
        started = await context.functions.call(
            "vln.navigate.start", instruction=context.task.instruction, options={}
        )
        job_id = started["job_id"]

        while not context.cancelled.is_set():
            state = await context.functions.call(
                "vln.navigate.status", job_id=job_id
            )
            context.output.append_jsonl("model/trace.jsonl", state)
            if state["state"] == "completed":
                context.output.add_artifact("model/trace.jsonl", "application/jsonl")
                await context.functions.call(
                    "nav.stop",
                    status="completed",
                    reason="VLN Job completed",
                    actor=self.name,
                )
                return
            if state["state"] in {"failed", "cancelled"}:
                await context.functions.call(
                    "nav.stop",
                    status="failed",
                    reason=state.get("error", f"VLN Job {state['state']}"),
                    actor=self.name,
                )
                return
            await asyncio.sleep(self.poll_interval_s)

        # Domain shutdown owns service cleanup after cancellation is announced.
        return

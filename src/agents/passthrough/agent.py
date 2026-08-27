from __future__ import annotations

from harness.runtime import NavContext


class PassthroughVLNAgent:
    """Pass each benchmark goal unchanged to one complete VLN job."""

    required_tools = frozenset(
        {
            "vln.navigate.task",
            "nav.goal.finish",
        }
    )

    async def run(self, context: NavContext) -> None:
        context.output.record(
            {
                "agent": type(self).__name__,
                "mode": "vln_passthrough",
                "required_tools": sorted(self.required_tools),
            }
        )
        instruction = context.task.instruction
        while True:
            status = await context.vln.navigate_task(instruction)
            if status["state"] != "succeeded":
                await context.nav.stop("failed", status.get("reason", "VLN job failed"))
                return
            transition = await context.nav.finish_goal(
                "completed", status.get("reason", "")
            )
            if transition["done"]:
                await context.nav.stop("completed", "all navigation goals completed")
                return
            instruction = transition["goal"]["instruction"]

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from domain.io import json_value
from domain.modules import Module


class MasterAgent(Module):
    """One-shot dispatcher used to validate the agent-to-subagent call graph."""

    def __init__(self) -> None:
        self._state = "created"

    def mount(self) -> None:
        self.context.register.register_reference(
            self.context.name,
            "master_agent.state",
            lambda: self._state,
            description="Master-agent lifecycle state.",
        )
        self.context.metadata.update(
            {"role": "master_agent", "policy": "instruction_type_stub"}
        )

    def run(self) -> None:
        self._state = "dispatching"
        task = dict(self.context.episode.instruction)
        function, arguments, can_complete_episode = select_capability(task)
        result = self.context.register.call(
            self.context.name,
            function,
            arguments,
        )
        record = {
            "task": task,
            "function": function,
            "arguments": arguments,
            "result": result,
        }
        self.context.output.write_json("dispatch.json", json_value(record))

        subtask_status = result.get("status") if isinstance(result, Mapping) else "failed"
        if subtask_status == "failed":
            terminal_status = "failed"
        elif subtask_status == "completed" and can_complete_episode:
            terminal_status = "completed"
        else:
            terminal_status = "incomplete"
        reason = (
            str(result.get("reason", "subtask returned no reason"))
            if isinstance(result, Mapping)
            else "subtask returned an invalid result"
        )
        self._state = "stopping"
        self.context.register.call(
            self.context.name,
            "env.stop",
            {
                "status": terminal_status,
                "reason": f"{function}: {reason}",
                "actor": self.context.name,
            },
        )
        self._state = "finished"


def select_capability(
    task: Mapping[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    task_type = str(task.get("type", ""))
    instruction = task.get("instruction")
    if task_type == "instruction" and isinstance(instruction, str) and instruction.strip():
        return "agent_vln.run", {"instruction": instruction}, True
    if task_type in {"target_text", "target_img"}:
        return "agent_objnav.run", {"target": dict(task)}, True
    if task_type in {"local", "target_pose"}:
        target_pose = task.get("target_pose", task)
        return "agent_local.run", {"target_pose": target_pose}, True
    focus = instruction if isinstance(instruction, str) else task_type
    return "agent_desc.run", {"focus": focus}, task_type == "describe"

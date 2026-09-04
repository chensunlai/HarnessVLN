from __future__ import annotations

import base64
import io
import json
import math
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modules.navigation_subagent import NavigationSubagent


PROMPT = """You control one short indoor VLN subtask. Follow the route instruction
from egocentric RGB-D observations, then return control to the master agent when the
local route is complete or materially contradicts the scene. A forward action moves
about 0.25 m and a turn action rotates about 15 degrees.

Policy:
- Parse the instruction into ordered route clauses. Preserve completed clauses in the
  progress ledger and work only on the first incomplete clause.
- The first image has the true starting heading. A route may begin behind the agent;
  scan systematically when needed and do not alternate directions repeatedly.
- Landmarks mentioned later need not be visible yet. Refuse only after reaching the
  expected transition and deliberately checking the relevant view, or after a clear
  geometric or semantic contradiction. Ordinary uncertainty calls for observation.
- Use image labels, pose, travelled distance, action counts, collision feedback, and
  the current 3x3 depth grid. Long-term views are uniformly sampled; recent views are
  consecutive and preserve local turn direction; the last image is always current.
- During a scan, compare the headings attached to recent views. Once an opening has
  been found, turn by the shortest signed heading change and advance. Do not revisit
  the same wall headings or alternate left and right while remaining in one place.
- Submit 1-4 atomic actions at a time. Use long batches only on a clearly open path;
  use short batches at doors, corners, furniture, and the endpoint. Never repeat a
  blocked forward action without changing heading.
- Visually servo while moving: center the next doorway, corridor, or route landmark
  before a forward batch. If it stays on one side of successive views, make a small
  corrective turn toward it instead of continuing on a diagonal. A requested left or
  right turn ends when the intended passage is centered, not when it first appears.
- Seeing the final landmark from afar is directional evidence, not arrival. Complete
  every ordered clause and approach the described stopping place to about one meter.
  Verify the named spatial relation literally: an endpoint still off to one side is
  usually guidance for the next correction, not proof that you occupy that endpoint.
- To complete, first call finish with confirm=false. Recheck the unchanged current
  image and ledger, then call finish with confirm=true. Any movement cancels it.
- Return exactly one native function call per response and no prose.
"""

PROGRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "current_place": {"type": "string", "maxLength": 160},
        "completed_steps": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "maxItems": 8,
        },
        "next_step": {"type": "string", "maxLength": 200},
        "decision": {"type": "string", "maxLength": 240},
    },
    "required": ["current_place", "completed_steps", "next_step", "decision"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "type": "function",
        "name": "agent_vln_act",
        "description": "Execute one short ordered batch and update route progress.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["forward", "turn_left", "turn_right"],
                    },
                    "minItems": 1,
                    "maxItems": 4,
                },
                "progress": PROGRESS_SCHEMA,
            },
            "required": ["actions", "progress"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "agent_vln_finish",
        "description": "Complete or refuse this local route and return to the master.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["completed", "refused"],
                },
                "reason": {"type": "string", "minLength": 1, "maxLength": 300},
                "confirm": {"type": "boolean"},
                "progress": PROGRESS_SCHEMA,
                "endpoint": {
                    "type": "object",
                    "properties": {
                        "visible": {"type": "boolean"},
                        "near": {"type": "boolean"},
                        "description": {"type": "string", "maxLength": 240},
                    },
                    "required": ["visible", "near", "description"],
                    "additionalProperties": False,
                },
            },
            "required": ["status", "reason", "confirm", "progress", "endpoint"],
            "additionalProperties": False,
        },
    },
]


class AgentVLNModule(NavigationSubagent):
    """Train-free visual controller for one bounded language-guided route."""

    function_name = "agent_vln.run"
    description = (
        "Execute a locally bounded natural-language route with RGB-D evidence. "
        "Refuse only when the expected route materially conflicts with the scene."
    )
    parameters = {
        "type": "object",
        "properties": {"instruction": {"type": "string", "minLength": 1}},
        "required": ["instruction"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        model: str = "gpt-5.6-terra",
        *,
        reasoning_effort: str = "high",
        max_iterations: int = 48,
        max_actions: int = 120,
        max_stationary_turns: int = 24,
        image_memory: int = 6,
        recent_images: int = 2,
        image_detail: str = "high",
        minimum_travel_m: float = 0.0,
        api_timeout_s: float = 180.0,
        model_retries: int = 2,
        retry_backoff_s: float = 2.0,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        if not model.strip() or reasoning_effort not in {
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise ValueError("model and reasoning_effort must be valid")
        if min(
            max_iterations,
            max_actions,
            max_stationary_turns,
            image_memory,
            recent_images,
        ) < 1:
            raise ValueError("iteration, action, and image limits must be positive")
        if recent_images > image_memory:
            raise ValueError("recent_images cannot exceed image_memory")
        if image_detail not in {"low", "high", "auto"} or minimum_travel_m < 0:
            raise ValueError("image detail or minimum travel is invalid")
        if api_timeout_s <= 0 or model_retries < 0 or retry_backoff_s < 0:
            raise ValueError("retry settings must be non-negative")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_iterations = max_iterations
        self.max_actions = max_actions
        self.max_stationary_turns = max_stationary_turns
        self.image_memory = image_memory
        self.recent_images = recent_images
        self.image_detail = image_detail
        self.minimum_travel_m = minimum_travel_m
        self.api_timeout_s = api_timeout_s
        self.model_retries = model_retries
        self.retry_backoff_s = retry_backoff_s
        self._client = client
        self._owns_client = client is None

    def mount(self) -> None:
        super().mount()
        self.context.metadata.update(
            {
                "policy": "responses_vln_loop",
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "max_actions_per_call": 4,
                "max_stationary_turns": self.max_stationary_turns,
                "image_memory": self.image_memory,
                "recent_images": self.recent_images,
            }
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None

    def execute(self, instruction: str, **_: Any) -> Mapping[str, Any]:
        state = _RouteState(
            instruction,
            self.context.output.path("trace.jsonl"),
            self.image_detail,
        )
        try:
            state.observe(self.observe())
            if state.frames[-1].get("image") is None:
                result = state.result("refused", "environment has no RGB observation")
            else:
                result = self._navigate(state)
        except Exception as error:
            result = state.result("failed", f"{type(error).__name__}: {error}")
        self.context.output.write_json("history.json", state.history())
        self.context.output.write_json("result.json", result)
        return result

    def _navigate(self, state: "_RouteState") -> dict[str, Any]:
        feedback = "Begin by grounding the first instruction clause."
        pending_revision: int | None = None
        for iteration in range(1, self.max_iterations + 1):
            if self.context.cancelled.is_set():
                return state.result("failed", "domain cancelled")
            if state.stationary_turns >= self.max_stationary_turns:
                return state.result(
                    "refused",
                    "no executable continuation found after a full local scan",
                )
            response = self._response(
                state.request(
                    iteration,
                    feedback,
                    self.image_memory,
                    self.recent_images,
                ),
                awaiting_confirmation=pending_revision is not None,
                action_available=len(state.actions) < self.max_actions,
            )
            calls = [item for item in response.output if item.type == "function_call"]
            if len(calls) != 1:
                raise ValueError(
                    f"model returned {len(calls)} function calls; expected one"
                )
            call = calls[0]
            try:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
            except (TypeError, json.JSONDecodeError, ValueError) as error:
                feedback = f"Invalid function arguments: {error}"
                state.trace(iteration, response, call, feedback)
                continue

            progress = _progress(arguments.get("progress"))
            if progress is None:
                feedback = "A complete structured progress ledger is required."
            elif call.name == "agent_vln_act":
                pending_revision = None
                feedback = self._act(state, arguments.get("actions"), progress)
            elif call.name == "agent_vln_finish":
                result, feedback, pending_revision = self._finish(
                    state, arguments, progress, pending_revision
                )
                state.trace(iteration, response, call, feedback)
                if result is not None:
                    return result
                continue
            else:
                feedback = f"Unknown function {call.name!r}."
            state.trace(iteration, response, call, feedback)
        return state.result("failed", "agent_vln iteration limit reached")

    def _act(
        self,
        state: "_RouteState",
        actions: Any,
        progress: Mapping[str, Any],
    ) -> str:
        allowed = {"forward", "turn_left", "turn_right"}
        if (
            not isinstance(actions, list)
            or not 1 <= len(actions) <= 4
            or any(action not in allowed for action in actions)
        ):
            return "Choose one to four valid atomic actions."
        remaining = self.max_actions - len(state.actions)
        if remaining < 1:
            return "Action budget is exhausted; finish completed or refused now."
        actions = actions[:remaining]

        state.progress = dict(progress)
        outcomes = []
        for action in actions:
            before = _position(state.observation)
            native = self.context.register.call(
                self.context.name, "env.step", {"action": action}
            )
            state.actions.append(action)
            state.revision += 1
            state.observe(self.observe())
            after = _position(state.observation)
            moved = (
                math.dist(before, after)
                if before is not None
                and after is not None
                and len(before) == len(after)
                else None
            )
            if moved is not None:
                state.travelled_m += moved
            blocked = action == "forward" and moved is not None and moved < 0.05
            if action.startswith("turn_"):
                state.stationary_turns += 1
            elif moved is None or moved >= 0.05:
                state.stationary_turns = 0
            outcomes.append(
                {
                    "action": action,
                    "translation_m": round(moved, 3) if moved is not None else None,
                    "blocked": blocked,
                }
            )
            state.last_actions = outcomes
            if isinstance(native, Mapping) and native.get("terminal"):
                return "The environment became terminal during the action batch."
            if blocked:
                break
        if len(state.actions) >= self.max_actions:
            return "Action budget is now exhausted; finish completed or refused now."
        return "Action batch executed. Use the new view and collision feedback."

    def _finish(
        self,
        state: "_RouteState",
        arguments: Mapping[str, Any],
        progress: Mapping[str, Any],
        pending_revision: int | None,
    ) -> tuple[dict[str, Any] | None, str, int | None]:
        status = arguments.get("status")
        reason = str(arguments.get("reason", "")).strip()
        endpoint = arguments.get("endpoint")
        if status not in {"completed", "refused"} or not reason:
            return None, "Finish requires a valid status and reason.", None
        if not isinstance(endpoint, Mapping):
            return None, "Finish requires endpoint evidence.", None
        state.progress = dict(progress)
        if status == "refused":
            return state.result("refused", reason, endpoint=endpoint), "refused", None

        errors = []
        if state.travelled_m + 0.01 < self.minimum_travel_m:
            errors.append(
                f"travelled {state.travelled_m:.2f} m; "
                f"minimum is {self.minimum_travel_m:.2f} m"
            )
        if not progress["completed_steps"]:
            errors.append("no completed instruction clauses")
        if not endpoint.get("visible") or not endpoint.get("near"):
            errors.append("endpoint is not both visible and near")
        if errors:
            return None, "Completion rejected: " + "; ".join(errors) + ".", None
        if not arguments.get("confirm"):
            return (
                None,
                "Arrival candidate recorded. Recheck the unchanged image and progress, "
                "then confirm or continue moving.",
                state.revision,
            )
        if pending_revision != state.revision:
            return None, "Submit finish with confirm=false at this location first.", None
        return state.result("completed", reason, endpoint=endpoint), "completed", None

    def _response(
        self,
        input_items: list[dict[str, Any]],
        *,
        awaiting_confirmation: bool,
        action_available: bool,
    ) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(timeout=self.api_timeout_s, max_retries=0)
        request = {
            "model": self.model,
            "instructions": PROMPT,
            "input": input_items,
            "tools": _tools(awaiting_confirmation, action_available),
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "reasoning": {"effort": self.reasoning_effort, "summary": "auto"},
            "store": False,
        }
        for attempt in range(self.model_retries + 1):
            try:
                return self._client.responses.create(**request)
            except Exception:
                if attempt >= self.model_retries:
                    raise
                time.sleep(self.retry_backoff_s * (2**attempt))
        raise AssertionError("unreachable")


class _RouteState:
    def __init__(self, instruction: str, trace_path: Path, image_detail: str) -> None:
        self.instruction = " ".join(instruction.split())
        self.trace_path = trace_path
        self.image_detail = image_detail
        self.observation: Mapping[str, Any] = {}
        self.frames: list[dict[str, Any]] = []
        self.actions: list[str] = []
        self.last_actions: list[dict[str, Any]] = []
        self.progress: dict[str, Any] = {
            "current_place": "start",
            "completed_steps": [],
            "next_step": "ground the first route clause",
            "decision": "not started",
        }
        self.travelled_m = 0.0
        self.stationary_turns = 0
        self.revision = 0
        self.events: list[dict[str, Any]] = []

    def observe(self, observation: Mapping[str, Any]) -> None:
        self.observation = observation
        channels = observation.get("channels")
        channels = channels if isinstance(channels, Mapping) else {}
        rgb = channels.get("rgb")
        self.frames.append(
            {
                "step": len(self.actions),
                "pose": _compact(observation.get("pose")),
                "image": _jpeg(rgb) if rgb is not None else None,
            }
        )

    def request(
        self,
        iteration: int,
        feedback: str,
        image_limit: int,
        recent_limit: int,
    ) -> list[dict[str, Any]]:
        channels = self.observation.get("channels")
        channels = channels if isinstance(channels, Mapping) else {}
        state = {
            "route_instruction": self.instruction,
            "iteration": iteration,
            "action_count": len(self.actions),
            "travelled_m": round(self.travelled_m, 3),
            "pose": _compact(self.observation.get("pose")),
            "depth_grid_m": _depth_grid(channels.get("depth")),
            "last_actions": self.last_actions,
            "action_counts": dict(Counter(self.actions)),
            "trailing_turns": _trailing_turns(self.actions),
            "stationary_turns": self.stationary_turns,
            "recent_headings": _recent_headings(self.frames),
            "progress": self.progress,
            "tool_feedback": feedback,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": "Current route state:\n"
                + json.dumps(state, ensure_ascii=True, separators=(",", ":")),
            }
        ]
        indices = _memory_indices(len(self.frames), image_limit, recent_limit)
        for order, index in enumerate(indices):
            frame = self.frames[index]
            if index == len(self.frames) - 1:
                role = "current"
            elif index >= len(self.frames) - recent_limit:
                role = "recent"
            else:
                role = "long_term"
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Uniform trajectory view {order + 1}/{len(indices)}: "
                        f"step={frame['step']}, role={role}, pose={frame['pose']}"
                    ),
                }
            )
            if frame["image"] is not None:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": frame["image"],
                        "detail": self.image_detail,
                    }
                )
        return [{"role": "user", "content": content}]

    def trace(self, iteration: int, response: Any, call: Any, feedback: str) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        try:
            arguments: Any = json.loads(call.arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = str(call.arguments)
        event = {
            "iteration": iteration,
            "response_id": getattr(response, "id", None),
            "usage": _compact(usage),
            "reasoning": _reasoning(response.output),
            "function": call.name,
            "arguments": arguments,
            "feedback": feedback,
            "step": len(self.actions),
            "travelled_m": round(self.travelled_m, 3),
        }
        self.events.append(event)
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")

    def result(
        self,
        status: str,
        reason: str,
        *,
        endpoint: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "reason": reason,
            "evidence": {
                "instruction": self.instruction,
                "progress": self.progress,
                "endpoint": dict(endpoint or {}),
                "action_count": len(self.actions),
                "action_counts": dict(Counter(self.actions)),
                "travelled_m": round(self.travelled_m, 3),
                "stationary_turns": self.stationary_turns,
                "observation_count": len(self.frames),
                "last_actions": self.last_actions,
            },
            "final_pose": _compact(self.observation.get("pose")),
        }

    def history(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "actions": self.actions,
            "travelled_m": round(self.travelled_m, 3),
            "progress": self.progress,
            "frames": [
                {"step": item["step"], "pose": item["pose"]}
                for item in self.frames
            ],
            "events": self.events,
        }


def _progress(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    keys = ("current_place", "completed_steps", "next_step", "decision")
    completed = value.get("completed_steps")
    if (
        not all(key in value for key in keys)
        or not isinstance(completed, list)
        or not all(isinstance(item, str) for item in completed)
        or not all(
            isinstance(value.get(key), str)
            for key in ("current_place", "next_step", "decision")
        )
    ):
        return None
    return {key: value[key] for key in keys}


def _tools(
    awaiting_confirmation: bool, action_available: bool = True
) -> list[dict[str, Any]]:
    tools = json.loads(json.dumps(TOOLS))
    if awaiting_confirmation:
        finish = tools[1]
        finish["description"] = (
            "Confirm the stationary arrival candidate, or refuse after rechecking it."
        )
        finish["parameters"]["properties"]["confirm"] = {
            "type": "boolean",
            "enum": [True],
        }
    return tools if action_available else [tools[1]]


def _uniform_indices(size: int, limit: int) -> list[int]:
    if limit == 1:
        return [size - 1]
    if size <= limit:
        return list(range(size))
    return sorted(
        {round(index * (size - 1) / (limit - 1)) for index in range(limit)}
    )


def _memory_indices(size: int, limit: int, recent: int) -> list[int]:
    if size <= limit:
        return list(range(size))
    recent = min(recent, limit, size)
    recent_indices = list(range(size - recent, size))
    history_limit = limit - recent
    if history_limit < 1:
        return recent_indices
    history_size = size - recent
    return _uniform_indices(history_size, history_limit) + recent_indices


def _recent_headings(frames: list[Mapping[str, Any]], limit: int = 8) -> list[Any]:
    headings = []
    for frame in frames[-limit:]:
        pose = frame.get("pose")
        headings.append(pose.get("heading_degrees") if isinstance(pose, Mapping) else None)
    return headings


def _position(observation: Mapping[str, Any]) -> tuple[float, ...] | None:
    pose = observation.get("pose")
    value = (
        pose.get("position", pose.get("gps")) if isinstance(pose, Mapping) else pose
    )
    tolist = getattr(value, "tolist", None)
    value = tolist() if callable(tolist) else value
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _trailing_turns(actions: list[str]) -> int:
    count = 0
    for action in reversed(actions):
        if action == "forward":
            break
        count += 1
    return count


def _depth_grid(depth: Any) -> list[list[float | None]]:
    if depth is None:
        return []
    import numpy as np

    array = np.asarray(depth, dtype=np.float32).squeeze()
    if array.ndim != 2 or not array.size:
        return []
    grid = []
    for row in np.array_split(array, 3, axis=0):
        values = []
        for cell in np.array_split(row, 3, axis=1):
            finite = cell[np.isfinite(cell) & (cell > 0)]
            values.append(round(float(np.median(finite)), 2) if finite.size else None)
        grid.append(values)
    return grid


def _jpeg(rgb: Any) -> str:
    import numpy as np
    from PIL import Image

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB observation must have shape HxWx3 or HxWx4")
    stream = io.BytesIO()
    Image.fromarray(array[:, :, :3].astype(np.uint8), "RGB").save(
        stream, format="JPEG", quality=85
    )
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()


def _reasoning(output: Any) -> str | None:
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        text = " ".join(
            str(getattr(part, "text", "")).strip()
            for part in getattr(item, "summary", ())
            if str(getattr(part, "text", "")).strip()
        )
        if text:
            return text
    return None


def _compact(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _compact(item)
            for key, item in value.items()
            if key not in {"rgb", "depth"}
        }
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    if hasattr(value, "tolist"):
        return _compact(value.tolist())
    if hasattr(value, "model_dump"):
        return _compact(value.model_dump(mode="json"))
    return str(value)

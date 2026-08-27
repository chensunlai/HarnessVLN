from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.runtime import NavContext
from harness.tool_bus import ToolSpec


DEFAULT_INSTRUCTIONS = """You are the decision core of a navigation agent.
Keep control of the complete benchmark goal with tool calls. The benchmark instruction
is the global objective and must never be copied or paraphrased as one VLN instruction.

For a route goal, first turn the global instruction into an ordered mental checklist.
Advance only the earliest unfinished clause. Never jump to a later landmark merely
because a similar object is visible; visually confirm that every preceding room, turn,
and landmark has been traversed before using the final landmark as evidence of arrival.

Before every vln.navigate.local call, use nav.observe and select exactly one traversable
target visible in the latest RGB image. Give one short sentence with an explicit local
stopping point. Prefer the farthest distinctive reachable anchor in the current view,
typically a visible object, wall, or floor region beyond a doorway. For example, "Go
through the doorway into the visible tiled hall and stop by its far wall." Do not use
the doorway threshold or "just beyond it" as a routine stop point. Never mention unseen
rooms or multiple route segments. Always set max_steps. For a language route, use the
schema maximum for a doorway, corridor, or room transition; reserve 4-8 steps for the
final nearby landmark. For object search, use about 8 steps to explore a passage and 4
to approach a visible candidate. The call blocks; afterwards observe again. A
limit_reached result is normal bounded progress, while failed indicates a real error.

Use local VLN calls as the main navigation strategy. Direct moves are only for brief
inspection, alignment, or final approach. Call one tool per response except that two
to four consecutive nav.move.discrete calls may be batched. Re-observe after every
movement batch or VLN call. Never batch any other tool.

When searching for the next visible passage, never spend one response on a single
15-degree inspection turn. Batch exactly four turns in the same direction in one
response, then observe once. Use one or two turns only for fine alignment to an already
visible target.

For an object goal, explore distinct passages systematically using visual landmarks
and pose history. Match the exact requested category and reject related but different
furniture or objects. Before approaching, compare a candidate against the closest
alternative object categories. Spend at most one local approach call on one candidate
before deciding from a fresh view; repeated approaches from the same area do not add
category evidence. If it remains ambiguous, mark it rejected and explore a different
passage. Architectural surfaces such as wall paneling, trim, and room doors are not
object instances. Seeing a candidate is not completion. Align and approach it, then
use the meter-valued depth grid from a fresh observation: the grid cell occupied by the
target should normally be within about 1 meter, and the target should occupy a
substantial part of that cell. A nearby wall or floor in another cell is not evidence.
Do not finish immediately after blocked motion or from an ambiguous single view.

Finish with a spatial safety margin. For a route landmark or goal area, move toward
its center or closest interior navigable point rather than stopping at its near edge.
If the last local call stopped on a boundary, observe and make one final 1-4 step
approach before nav.goal.finish. For an object goal, keep the exact target clearly in
view during this final approach and do not pass it.

Use observation position and heading as a compact route ledger. Do not issue the same
local target again from a nearby pose, revisit a completed room transition, or perform
repeated full scans at one junction. If progress returns near an earlier pose, choose
a visibly different unexplored exit. Prefer a few decisive local calls over many short
calls and inspection turns.

Finish every current goal with nav.goal.finish and inspect its accepted/done result.
Call nav.stop with status "completed" only after the final goal was accepted and done.
Use status "failed" for genuine failure; never use "success" as a status. Never end a
turn with plain text."""


REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


class AgentToolPolicyError(ValueError):
    pass


class NormalAgent:
    """Minimal Responses API agent loop over Harness navigation tools."""

    def __init__(
        self,
        model: str,
        tools: Sequence[str],
        *,
        instructions: str = DEFAULT_INSTRUCTIONS,
        guidance: str = "",
        max_iterations: int = 80,
        max_actions_per_turn: int = 4,
        reasoning_effort: str | None = None,
        observation_image_detail: str = "high",
        max_navigation_actions: int = 240,
        model_retries: int = 3,
        retry_backoff_s: float = 2.0,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if not tools:
            raise ValueError("tools must not be empty")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 1 <= max_actions_per_turn <= 4:
            raise ValueError("max_actions_per_turn must be between 1 and 4")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning_effort: {reasoning_effort}")
        if observation_image_detail not in {"low", "high", "auto"}:
            raise ValueError(
                "observation_image_detail must be low, high, or auto"
            )
        if type(max_navigation_actions) is not int or max_navigation_actions <= 0:
            raise ValueError("max_navigation_actions must be positive")
        if model_retries < 0:
            raise ValueError("model_retries must not be negative")
        if retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must not be negative")
        self.model = model
        self.instructions = _join_instructions(instructions, guidance)
        self.max_iterations = max_iterations
        self.max_actions_per_turn = max_actions_per_turn
        self.reasoning_effort = reasoning_effort
        self.observation_image_detail = observation_image_detail
        self.max_navigation_actions = max_navigation_actions
        self.model_retries = model_retries
        self.retry_backoff_s = retry_backoff_s
        self.required_tools = frozenset(tools)
        forbidden_vln_tools = self.required_tools & {
            "vln.navigate.task",
            "vln.navigate.start",
            "vln.navigate.status",
            "vln.navigate.cancel",
        }
        if forbidden_vln_tools:
            names = ", ".join(sorted(forbidden_vln_tools))
            raise ValueError(
                "NormalAgent only supports blocking local VLN navigation; "
                f"remove: {names}"
            )
        if (
            "vln.navigate.local" in self.required_tools
            and "nav.observe" not in self.required_tools
        ):
            raise ValueError("vln.navigate.local requires nav.observe")
        self._client = client

    async def run(self, context: NavContext) -> None:
        trace_path = "components/agent.events.jsonl"
        context.output.record(
            {
                "agent": type(self).__name__,
                "mode": "free_agent_loop",
                "model": self.model,
                "max_iterations": self.max_iterations,
                "max_actions_per_turn": self.max_actions_per_turn,
                "reasoning_effort": self.reasoning_effort,
                "observation_image_detail": self.observation_image_detail,
                "max_navigation_actions": self.max_navigation_actions,
                "model_retries": self.model_retries,
                "retry_backoff_s": self.retry_backoff_s,
                "required_tools": sorted(self.required_tools),
                "model_trace": {
                    "schema_version": 1,
                    "format": "jsonl",
                    "path": trace_path,
                },
            }
        )
        model_tools, tool_names = _responses_tools(context.tools.specs)
        initial_input = {
            "role": "user",
            "content": "Navigate this task:\n"
            + _json(
                {
                    **_task_data(context),
                    "limits": {
                        "navigation_actions": self.max_navigation_actions,
                    },
                }
            ),
        }
        input_items: list[Any] = [initial_input]
        trace_sequence = 0
        response_count = 0
        usage: dict[str, int] = {}
        fresh_observation = False
        navigation_actions = 0
        final_goal_accepted = False
        require_observation = "nav.observe" in self.required_tools
        require_goal_finish = "nav.goal.finish" in self.required_tools
        local_step_cap = _local_step_cap(context.tools.specs)
        task_instructions = {_instruction_key(context.task.instruction)}

        def trace(event_type: str, **data: Any) -> None:
            nonlocal trace_sequence
            trace_sequence += 1
            context.output.event(
                {
                    "schema_version": 1,
                    "sequence": trace_sequence,
                    "time_unix": time.time(),
                    "type": event_type,
                    **data,
                }
            )

        trace(
            "agent.started",
            execution_id=context.execution_id,
            model=self.model,
            instructions=self.instructions,
            input=initial_input,
            tools=model_tools,
        )

        try:
            for iteration in range(1, self.max_iterations + 1):
                if context.cancelled.is_set():
                    trace("agent.cancelled", iteration=iteration)
                    return
                request: dict[str, Any] = {
                    "model": self.model,
                    "instructions": self.instructions,
                    "input": input_items,
                    "tools": model_tools,
                    "tool_choice": "required",
                    "parallel_tool_calls": self.max_actions_per_turn > 1,
                    "store": False,
                }
                if self.reasoning_effort is not None:
                    request["reasoning"] = {"effort": self.reasoning_effort}
                trace(
                    "model.request",
                    iteration=iteration,
                    input_item_count=len(input_items),
                    request={
                        key: _trace_data(value)
                        for key, value in request.items()
                        if key != "input"
                    },
                )
                try:
                    response = await self._create_response(request)
                except Exception as error:
                    trace(
                        "model.error",
                        iteration=iteration,
                        error={
                            "type": type(error).__name__,
                            "message": str(error),
                            "status_code": getattr(error, "status_code", None),
                        },
                    )
                    raise

                response_data = _trace_data(response)
                response_count += 1
                _accumulate_usage(usage, response_data)
                trace(
                    "model.response",
                    iteration=iteration,
                    response=response_data,
                )
                context.output.record(
                    {
                        "model_responses": response_count,
                        "usage": usage,
                    }
                )

                output = list(response.output)
                input_items.extend(output)
                calls = [item for item in output if item.type == "function_call"]
                if not calls:
                    trace(
                        "agent.terminal",
                        iteration=iteration,
                        status="failed",
                        reason="model returned no tool calls",
                    )
                    await context.nav.stop(
                        "failed", "model returned no tool calls"
                    )
                    return

                batch_error = _batch_error(
                    calls, tool_names, maximum=self.max_actions_per_turn
                )
                canonical_calls = [tool_names.get(call.name) for call in calls]
                if batch_error is None and all(
                    name == "nav.move.discrete" for name in canonical_calls
                ):
                    if require_observation and not fresh_observation:
                        batch_error = (
                            "nav.move.discrete requires a fresh nav.observe before "
                            "the movement batch"
                        )
                    elif navigation_actions + len(calls) > self.max_navigation_actions:
                        batch_error = (
                            "movement batch exceeds the remaining navigation action "
                            "budget: "
                            f"{self.max_navigation_actions - navigation_actions}"
                        )
                skip_remaining_moves: str | None = None
                for call in calls:
                    canonical_name: str | None = tool_names.get(call.name)
                    arguments: Any = call.arguments
                    compact_images = False
                    if batch_error is not None:
                        tool_output = _error_output("ToolBatchError", batch_error)
                    elif skip_remaining_moves is not None:
                        tool_output = _error_output(
                            "ToolBatchSkipped", skip_remaining_moves
                        )
                    else:
                        try:
                            if canonical_name is None:
                                raise ValueError(f"unknown model tool: {call.name}")
                            arguments = json.loads(call.arguments)
                            if not isinstance(arguments, dict):
                                raise ValueError("tool arguments must be a JSON object")
                            arguments = _normalize_tool_arguments(
                                canonical_name, arguments
                            )
                            if canonical_name == "vln.navigate.local":
                                if not fresh_observation:
                                    raise AgentToolPolicyError(
                                        "vln.navigate.local requires a fresh "
                                        "nav.observe after the most recent movement "
                                        "or VLN call"
                                    )
                                requested_steps = _validate_local_instruction(
                                    arguments,
                                    task_instructions,
                                    maximum=local_step_cap,
                                )
                                if (
                                    navigation_actions + requested_steps
                                    > self.max_navigation_actions
                                ):
                                    remaining = (
                                        self.max_navigation_actions
                                        - navigation_actions
                                    )
                                    raise AgentToolPolicyError(
                                        "local VLN call exceeds the remaining "
                                        "navigation action budget: "
                                        f"{remaining}"
                                    )
                                fresh_observation = False
                                compact_images = True
                            elif canonical_name == "nav.goal.finish":
                                if (
                                    arguments.get("status") == "completed"
                                    and require_observation
                                    and not fresh_observation
                                ):
                                    raise AgentToolPolicyError(
                                        "nav.goal.finish requires a fresh nav.observe "
                                        "after the most recent movement or VLN call"
                                    )
                            elif canonical_name == "nav.stop":
                                if (
                                    arguments.get("status") == "completed"
                                    and require_goal_finish
                                    and not final_goal_accepted
                                ):
                                    raise AgentToolPolicyError(
                                        "nav.stop completed requires an accepted final "
                                        "nav.goal.finish result"
                                    )
                            result = await context.tools.call(canonical_name, arguments)
                            if canonical_name == "nav.observe":
                                fresh_observation = True
                                compact_images = True
                            elif canonical_name == "nav.move.discrete":
                                fresh_observation = False
                                navigation_actions += 1
                                compact_images = True
                                stop_reason = _move_batch_stop_reason(result)
                                if stop_reason is not None:
                                    skip_remaining_moves = stop_reason
                            elif canonical_name == "vln.navigate.local":
                                navigation_actions += _navigation_steps(
                                    result, requested_steps
                                )
                            elif canonical_name == "nav.goal.finish":
                                fresh_observation = False
                                compact_images = True
                                final_goal_accepted = bool(
                                    isinstance(result, Mapping)
                                    and result.get("accepted") is True
                                    and result.get("done") is True
                                )
                                next_goal = (
                                    result.get("goal")
                                    if isinstance(result, Mapping)
                                    else None
                                )
                                if isinstance(next_goal, Mapping):
                                    final_goal_accepted = False
                                    task_instructions.add(
                                        _instruction_key(next_goal.get("instruction"))
                                    )
                            elif canonical_name == "nav.stop":
                                compact_images = True
                            tool_output = {"ok": True, "result": result}
                            if canonical_name in {
                                "nav.move.discrete",
                                "vln.navigate.local",
                            }:
                                tool_output["navigation_budget"] = {
                                    "used": navigation_actions,
                                    "remaining": self.max_navigation_actions
                                    - navigation_actions,
                                }
                        except Exception as error:
                            tool_output = _error_output(
                                type(error).__name__, str(error)
                            )

                    if compact_images:
                        input_items = _compact_observation_images(input_items)

                    model_input = _function_call_output(
                        call.call_id,
                        canonical_name,
                        tool_output,
                        image_detail=self.observation_image_detail,
                    )
                    input_items.append(model_input)
                    trace(
                        "tool.result",
                        iteration=iteration,
                        call_id=call.call_id,
                        provider_name=call.name,
                        tool_name=canonical_name,
                        arguments=arguments,
                        output=_trace_data(tool_output),
                        model_input=_trace_model_input(model_input),
                    )
                    if canonical_name == "nav.stop" and tool_output["ok"]:
                        trace(
                            "agent.terminal",
                            iteration=iteration,
                            status=arguments.get("status", "completed"),
                            reason=arguments.get("reason", ""),
                        )
                        return

            trace(
                "agent.terminal",
                iteration=self.max_iterations,
                status="failed",
                reason="agent iteration budget reached",
            )
            await context.nav.stop("failed", "agent iteration budget reached")
        finally:
            context.output.record(
                {
                    "model_responses": response_count,
                    "usage": usage,
                    "trace_events": trace_sequence,
                    "navigation_actions": navigation_actions,
                }
            )

    def _responses(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        responses = getattr(self._client, "responses", None)
        if responses is None:
            raise RuntimeError("the installed OpenAI SDK does not expose Responses API")
        return responses

    async def _create_response(self, request: Mapping[str, Any]) -> Any:
        for retry in range(self.model_retries + 1):
            try:
                return await self._responses().create(**request)
            except Exception as error:
                if retry >= self.model_retries or not _is_transient_model_error(error):
                    raise
                await asyncio.sleep(self.retry_backoff_s * 2**retry)
        raise AssertionError("unreachable model retry state")


def _batch_error(
    calls: Sequence[Any],
    tool_names: Mapping[str, str],
    *,
    maximum: int,
) -> str | None:
    if len(calls) > maximum:
        return f"model returned {len(calls)} tool calls; maximum is {maximum}"
    if len(calls) == 1:
        return None
    canonical_names = [tool_names.get(call.name) for call in calls]
    if any(name != "nav.move.discrete" for name in canonical_names):
        return "only consecutive nav.move.discrete calls may be batched"
    return None


def _local_step_cap(specs: Sequence[ToolSpec]) -> int:
    for spec in specs:
        if spec.name != "vln.navigate.local":
            continue
        properties = spec.input_schema.get("properties")
        max_steps = (
            properties.get("max_steps")
            if isinstance(properties, Mapping)
            else None
        )
        maximum = max_steps.get("maximum") if isinstance(max_steps, Mapping) else None
        if type(maximum) is not int or maximum <= 0:
            raise ValueError(
                "vln.navigate.local must expose a positive max_steps maximum"
            )
        return maximum
    return 0


def _validate_local_instruction(
    arguments: Mapping[str, Any],
    task_instructions: set[str],
    *,
    maximum: int,
) -> int:
    instruction = arguments.get("instruction")
    if not isinstance(instruction, str):
        raise AgentToolPolicyError("local VLN instruction must be a string")
    normalized = _instruction_key(instruction)
    if normalized in task_instructions:
        raise AgentToolPolicyError(
            "a local VLN instruction must describe one target visible in the latest "
            "observation, not copy the current benchmark goal"
        )
    if len(instruction) > 180:
        raise AgentToolPolicyError(
            "local VLN instruction must not exceed 180 characters"
        )
    if "\n" in instruction or ";" in instruction or re.search(
        r"\b(?:then|afterwards|subsequently)\b", normalized
    ):
        raise AgentToolPolicyError(
            "local VLN instruction must contain one visible route segment"
        )
    if len(re.findall(r"[.!?]+", instruction.strip())) > 1:
        raise AgentToolPolicyError("local VLN instruction must be one sentence")
    if re.search(r"\b(?:stop|stopping)\b", normalized) is None:
        raise AgentToolPolicyError(
            "local VLN instruction must state where to stop at the visible target"
        )
    max_steps = arguments.get("max_steps")
    if type(max_steps) is not int or not 1 <= max_steps <= maximum:
        raise AgentToolPolicyError(
            f"local max_steps must be an integer between 1 and {maximum}"
        )
    return max_steps


def _navigation_steps(result: Any, requested: int) -> int:
    steps = result.get("steps") if isinstance(result, Mapping) else None
    return steps if type(steps) is int and 0 <= steps <= requested else requested


def _move_batch_stop_reason(result: Any) -> str | None:
    if not isinstance(result, Mapping):
        return None
    if result.get("native_terminal") is True:
        return "environment reached a native terminal state; remaining moves skipped"
    motion = result.get("motion")
    if isinstance(motion, Mapping) and motion.get("blocked") is True:
        return (
            "movement was blocked; remaining moves skipped, observe before replanning"
        )
    return None


def _compact_observation_images(items: Sequence[Any]) -> list[Any]:
    compacted = list(items)
    for index, item in enumerate(compacted):
        if not isinstance(item, Mapping) or item.get("type") != "function_call_output":
            continue
        output = item.get("output")
        if not isinstance(output, list) or not any(
            isinstance(part, Mapping) and part.get("type") == "input_image"
            for part in output
        ):
            continue
        compacted[index] = {
            **dict(item),
            "output": [
                dict(part)
                for part in output
                if isinstance(part, Mapping) and part.get("type") != "input_image"
            ],
        }
    return compacted


def _join_instructions(instructions: str, guidance: str) -> str:
    base = instructions.strip()
    extra = guidance.strip()
    return f"{base}\n{extra}" if extra else base


def _error_output(error_type: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {"type": error_type, "message": message},
    }


def _is_transient_model_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    } or isinstance(error, (ConnectionError, TimeoutError))


def _normalize_tool_arguments(
    name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if name not in {"nav.goal.finish", "nav.stop"}:
        return arguments
    status = arguments.get("status")
    if not isinstance(status, str):
        return arguments
    normalized = status.strip().casefold()
    if normalized in {"success", "succeeded", "done"}:
        return {**arguments, "status": "completed"}
    if normalized in {"failure", "error", "unsuccessful"}:
        return {**arguments, "status": "failed"}
    return arguments


def _instruction_key(value: Any) -> str:
    return " ".join(value.split()).casefold() if isinstance(value, str) else ""


def _function_call_output(
    call_id: str,
    tool_name: str | None,
    tool_output: dict[str, Any],
    *,
    image_detail: str,
) -> dict[str, Any]:
    output: Any = _json(_model_tool_output(tool_name, tool_output))
    image_url = _observation_image_url(tool_name, tool_output)
    if image_url is not None:
        output = [
            {"type": "input_text", "text": output},
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": image_detail,
            },
        ]
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def _trace_model_input(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep model-facing feedback while representing inline images compactly."""
    result = _trace_data(value)
    output = result.get("output")
    if not isinstance(output, list):
        return result
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "input_image":
            continue
        image_url = item.get("image_url")
        if not isinstance(image_url, str) or not image_url.startswith("data:"):
            continue
        header, _, payload = image_url.partition(",")
        item["image_url"] = {
            "source": "nav.observe.channels.rgb",
            "media_type": header.removeprefix("data:").split(";", 1)[0],
            "encoded_bytes": len(payload),
        }
    return result


def _trace_data(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _trace_data(dump(mode="json"))
        except TypeError:
            return _trace_data(dump())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _trace_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_trace_data(item) for item in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    shape = getattr(value, "shape", None)
    if shape is not None:
        return _json_default(value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _trace_data(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return {"type": type(value).__name__}


def _accumulate_usage(total: dict[str, int], response: Any) -> None:
    if not isinstance(response, Mapping):
        return
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            total[name] = total.get(name, 0) + value


def _model_tool_output(
    tool_name: str | None, tool_output: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(tool_output)
    if tool_name != "nav.observe" or tool_output.get("ok") is not True:
        return value
    result = tool_output.get("result")
    if not isinstance(result, Mapping):
        return value
    channels = result.get("channels")
    if not isinstance(channels, Mapping):
        return value
    depth_summary = _depth_summary(
        channels.get("depth"), channels.get("depth_metadata")
    )
    if depth_summary is not None:
        value["sensor_summary"] = {"depth": depth_summary}
    return value


def _depth_summary(depth: Any, metadata: Any = None) -> dict[str, Any] | None:
    shape = tuple(getattr(depth, "shape", ()))
    if len(shape) == 3 and shape[2] == 1:
        depth = depth[:, :, 0]
        shape = shape[:2]
    if len(shape) != 2 or min(shape) <= 0:
        return None

    import numpy as np

    values = np.asarray(depth)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    height, width = values.shape
    center = values[
        height * 2 // 5 : height * 3 // 5,
        width * 2 // 5 : width * 3 // 5,
    ]
    grid: list[list[float | None]] = []
    for row in range(3):
        cells: list[float | None] = []
        for column in range(3):
            cell = values[
                height * row // 3 : height * (row + 1) // 3,
                width * column // 3 : width * (column + 1) // 3,
            ]
            cells.append(_finite_median(cell))
        grid.append(cells)
    summary = {
        "center": _finite_median(center),
        "grid": grid,
        "minimum": round(float(finite.min()), 4),
        "maximum": round(float(finite.max()), 4),
        "lower_is_nearer": True,
    }
    metric_range = _depth_metric_range(metadata)
    if metric_range is not None:
        minimum_m, maximum_m = metric_range
        summary["meters"] = {
            "center": _to_meters(summary["center"], minimum_m, maximum_m),
            "grid": [
                [_to_meters(value, minimum_m, maximum_m) for value in row]
                for row in grid
            ],
            "sensor_range": [minimum_m, maximum_m],
        }
    return summary


def _depth_metric_range(metadata: Any) -> tuple[float, float] | None:
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("encoding") != "linear_normalized":
        return None
    minimum = metadata.get("minimum_m")
    maximum = metadata.get("maximum_m")
    if (
        not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
        or not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or float(minimum) >= float(maximum)
    ):
        return None
    return float(minimum), float(maximum)


def _to_meters(
    value: float | None, minimum_m: float, maximum_m: float
) -> float | None:
    if value is None:
        return None
    return round(minimum_m + value * (maximum_m - minimum_m), 4)


def _finite_median(values: Any) -> float | None:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return round(float(np.median(finite)), 4)


def _observation_image_url(
    tool_name: str | None, tool_output: Mapping[str, Any]
) -> str | None:
    if tool_name != "nav.observe" or tool_output.get("ok") is not True:
        return None
    result = tool_output.get("result")
    if not isinstance(result, Mapping):
        return None
    channels = result.get("channels")
    if not isinstance(channels, Mapping):
        return None
    rgb = channels.get("rgb")
    if rgb is None:
        return None
    shape = tuple(getattr(rgb, "shape", ()))
    if len(shape) != 3 or shape[2] not in {3, 4} or min(shape) <= 0:
        return None
    if str(getattr(rgb, "dtype", "")) != "uint8":
        return None
    from PIL import Image

    image = Image.fromarray(rgb).convert("RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=80, optimize=True)
    payload = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _responses_tools(
    specs: Sequence[ToolSpec],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tools: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for spec in specs:
        model_name = spec.name.replace(".", "__")
        if model_name in names:
            raise ValueError(f"tool name collision after provider mapping: {spec.name}")
        names[model_name] = spec.name
        tools.append(
            {
                "type": "function",
                "name": model_name,
                "description": spec.description,
                "parameters": spec.input_schema,
                "strict": False,
            }
        )
    return tools, names


def _task_data(context: NavContext) -> dict[str, Any]:
    task = context.task
    return {
        "task_id": task.task_id,
        "scene_id": task.scene_id,
        "goal": {
            "goal_id": task.goal.goal_id,
            "instruction": task.goal.instruction,
            "modality": task.goal.modality,
            "public": dict(task.goal.public),
        },
        "public": dict(task.public),
    }


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    shape = getattr(value, "shape", None)
    if shape is not None:
        size = getattr(value, "size", None)
        tolist = getattr(value, "tolist", None)
        if isinstance(size, int) and size <= 64 and callable(tolist):
            return tolist()
        return {
            "type": "array",
            "shape": [int(item) for item in shape],
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
    return {"type": type(value).__name__}

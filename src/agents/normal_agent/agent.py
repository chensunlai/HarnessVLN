from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.runtime import NavContext
from harness.tool_bus import ToolSpec


DEFAULT_INSTRUCTIONS = """You are the decision core of a navigation agent.
Keep control of the task with tool calls. Delegate the main navigation attempt to a
VLN tool when one is available; direct atomic moves are only for short inspection or
correction and must not be the sole task strategy. Call one tool per response except
that you may emit two to four consecutive nav.move.discrete calls; they execute in
output order. Never batch observation, VLN, memory, goal-finish, or stop calls.
Finish each current goal with the goal-finish tool. Finish the complete task only
with the stop tool. For both tools use status "completed" after success and status
"failed" after failure; never use "success" as a status. Do not end a turn with
plain text. For an object-modality goal, after the VLN succeeds, observe the current
RGB image and use short atomic corrections when needed before finishing the goal.
Merely seeing the target object is not completion: center it, approach it until it
is clearly near, and re-observe between action batches before calling goal-finish.
When a depth summary is present, use its center and 3-by-3 grid together with the
reported depth metadata to verify proximity; lower normalized values are nearer.
Before finishing an object goal, attempt a forward correction toward the centered
candidate and inspect the action's motion.blocked feedback. Low depth at a blocked
surface does not prove success. A target is centered only when its body occupies the
middle third of the RGB image. Never finish after a blocked forward action: find a
new approach angle, then obtain at least two unblocked forward translations with
fresh observations after the most recent block before considering goal-finish."""


REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


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
        self.model_retries = model_retries
        self.retry_backoff_s = retry_backoff_s
        self.required_tools = frozenset(tools)
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
            "content": "Navigate this task:\n" + _json(_task_data(context)),
        }
        input_items: list[Any] = [initial_input]
        trace_sequence = 0
        response_count = 0
        usage: dict[str, int] = {}

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
                for call in calls:
                    canonical_name: str | None = tool_names.get(call.name)
                    arguments: Any = call.arguments
                    if batch_error is not None:
                        tool_output = _error_output("ToolBatchError", batch_error)
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
                            result = await context.tools.call(canonical_name, arguments)
                            tool_output = {"ok": True, "result": result}
                        except Exception as error:
                            tool_output = _error_output(type(error).__name__, str(error))

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
    depth_summary = _depth_summary(channels.get("depth"))
    if depth_summary is not None:
        value["sensor_summary"] = {"depth": depth_summary}
    return value


def _depth_summary(depth: Any) -> dict[str, Any] | None:
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
    return {
        "center": _finite_median(center),
        "grid": grid,
        "minimum": round(float(finite.min()), 4),
        "maximum": round(float(finite.max()), 4),
        "lower_is_nearer": True,
    }


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

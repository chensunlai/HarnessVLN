from __future__ import annotations

import asyncio
import base64
import inspect
import io
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from domain.modules import Module, ModuleContext


INSTRUCTIONS = """You are the only controller for an embodied indoor navigation task.
Follow the route instruction from the current first-person RGB observation. Use only
the provided tools and keep navigating until you have strong visual evidence that the
described endpoint has been reached.

Navigation policy:
- One move_forward advances about 0.25 m and one turn changes heading by 15 degrees.
  Treat cumulative travelled distance and GPS displacement as odometry, not as a goal
  signal.
- nav_act executes 1 to 4 atomic actions in order. Batch repeated forward or turn
  actions only when the visible path is clearly safe; use shorter batches near doors,
  corners, stairs, and obstacles.
- Turn to localize landmarks and preserve the instruction's ordered route clauses.
- Treat room transitions and turns as ordered prerequisites. A visually matching final
  landmark is not the destination if earlier clauses such as exiting the starting
  room, entering a hallway, or taking a specified turn have not been verified.
- Historical images are sampled uniformly over the trajectory and may have large time
  gaps. Use their step and pose fields to order them; the final image is always the
  current observation. Use this history to avoid scanning the same headings repeatedly.
  Once a plausible direction is found, make forward progress instead of oscillating
  between left and right views. Treat a search_loop_risk diagnostic as a direct signal
  to stop rescanning and choose the best previously observed open route.
- Decompose the route instruction into ordered clauses yourself. Every nav_act call
  must carry a navigation_memory ledger whose completed_route is your authoritative,
  compact summary of the instruction clauses already verified from visual evidence.
  Keep completed clauses stable, update the current place, and name exactly the next
  uncompleted clause or landmark. Do not replace verified progress with a new guess.
- Use movement feedback, pose change, heading, and depth samples to detect blocked or
  mistaken motion. Do not repeat a blocked forward action.
- R2R endpoint landmarks are often visible well before the stopping location. Seeing
  the named bed, chair, doorway, or rug is evidence for direction, not evidence of
  arrival. Approach the specified place at human standing distance and complete all
  route clauses. Unless the instruction explicitly describes an immediate stop, a
  normal route requires materially more than one or two forward steps.
- Call nav_observe when another look at the unchanged state is useful.
- Call nav_stop with status "completed" only after every ordered route clause is
  complete and the endpoint is visibly near. First submit the evidence with
  confirm=false. Recheck the unchanged image and ledger after the verification
  response, then use confirm=true only if all evidence still holds. Any nav_act
  cancels the pending confirmation. The environment decides benchmark success.
- Every response must contain exactly one native function call. Do not answer in text.
"""


class NormalAgent(Module):
    """Small Responses API loop over observation, atomic action, and stop tools."""

    name = "agent"
    required_functions = frozenset({"nav.observe", "nav.act", "nav.stop"})

    def __init__(
        self,
        model: str = "gpt-5.6-terra",
        tools: Sequence[str] = ("env.observe", "env.step"),
        *,
        reasoning_effort: str = "high",
        max_iterations: int = 80,
        max_actions: int = 160,
        minimum_travel_m: float = 0.0,
        image_memory_turns: int = 6,
        history_checkpoints: int = 24,
        image_detail: str = "high",
        model_retries: int = 2,
        retry_backoff_s: float = 2.0,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("reasoning_effort must be low, medium, high, or xhigh")
        if max_iterations < 1 or max_actions < 1:
            raise ValueError("iteration and action limits must be positive")
        if minimum_travel_m < 0:
            raise ValueError("minimum_travel_m must not be negative")
        if image_memory_turns < 1 or history_checkpoints < 1:
            raise ValueError("memory limits must be positive")
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be low, high, or auto")
        if model_retries < 0 or retry_backoff_s < 0:
            raise ValueError("retry settings must not be negative")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_iterations = max_iterations
        self.max_actions = max_actions
        self.minimum_travel_m = minimum_travel_m
        self.image_memory_turns = image_memory_turns
        self.history_checkpoints = history_checkpoints
        self.image_detail = image_detail
        self.model_retries = model_retries
        self.retry_backoff_s = retry_backoff_s
        required_tools = frozenset(str(name) for name in tools) | {"env.stop"}
        if not {"env.observe", "env.step"} <= required_tools:
            raise ValueError("NormalAgent requires env.observe and env.step")
        self.tools = required_tools
        self._client = client
        self._owns_client = client is None

    def close(self) -> None:
        if self._client is not None:
            asyncio.run(self._close_client())

    async def _close_client(self) -> None:
        if not self._owns_client or self._client is None:
            return
        client, self._client = self._client, None
        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def run(self) -> None:
        asyncio.run(self._run_and_close())

    async def _run_and_close(self) -> None:
        try:
            await self._run(self.context)
        finally:
            await self._close_client()

    async def _run(self, module_context: ModuleContext) -> None:
        context = _FabricContext(module_context, self.tools)
        context.output.set_metadata(
            agent=type(self).__name__,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_atomic_actions_per_call=4,
            max_actions=self.max_actions,
            minimum_travel_m=self.minimum_travel_m,
            image_memory_turns=self.image_memory_turns,
            image_sampling="uniform_over_trajectory",
            history_checkpoints=self.history_checkpoints,
            structured_navigation_memory=True,
            stop_confirmation=True,
        )
        tools, model_names = _model_tools(context.functions.specs)
        observation = await context.functions.call("nav.observe")
        action_history: list[str] = []
        current_message = _observation_message(
            context.task.instruction,
            observation,
            action_history,
            self.image_detail,
        )
        turn_history: list[tuple[list[Any], dict[str, Any], int]] = []
        all_events: list[dict[str, Any]] = []
        state_revision = 0
        stop_candidate_revision: int | None = None

        def save_history() -> None:
            context.output.write_json(
                "model/history.json",
                {
                    "summary": _history_data(
                        all_events, self.history_checkpoints, action_history
                    ),
                    "turns": all_events,
                },
            )
            context.output.add_artifact("model/history.json", "application/json")

        for iteration in range(1, self.max_iterations + 1):
            if context.cancelled.is_set():
                save_history()
                return
            input_state_revision = state_revision
            input_items, sampled_history_indices = _memory_input(
                turn_history,
                current_message,
                all_events,
                self.history_checkpoints,
                action_history,
                self.image_memory_turns,
                input_state_revision,
            )
            request = {
                "model": self.model,
                "instructions": INSTRUCTIONS,
                "input": input_items,
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "reasoning": {
                    "effort": self.reasoning_effort,
                    "summary": "auto",
                },
                "store": False,
            }
            try:
                response = await self._create_response(request)
            except Exception as error:
                context.output.append_jsonl(
                    "model/trace.jsonl",
                    {
                        "iteration": iteration,
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                )
                context.output.add_artifact("model/trace.jsonl", "application/jsonl")
                save_history()
                raise
            calls = [item for item in response.output if item.type == "function_call"]

            if len(calls) != 1:
                context.output.append_jsonl(
                    "model/trace.jsonl", _response_record(iteration, response, calls)
                )
                context.output.add_artifact("model/trace.jsonl", "application/jsonl")
                save_history()
                await context.functions.call(
                    "nav.stop",
                    status="failed",
                    reason=f"model returned {len(calls)} function calls; expected one",
                    actor=self.name,
                )
                return

            call = calls[0]
            function_name = model_names.get(call.name)
            try:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("function arguments must be an object")
            except (TypeError, json.JSONDecodeError, ValueError) as error:
                arguments = {}
                result: Any = {"error": f"invalid arguments: {error}"}
            else:
                if function_name is None:
                    result = {"error": f"unknown function: {call.name}"}
                elif function_name == "nav.act" and (
                    memory_error := _navigation_memory_error(arguments)
                ):
                    stop_candidate_revision = None
                    result = {"error": memory_error}
                elif function_name == "nav.stop" and self._premature_stop(
                    arguments, observation
                ):
                    stop_candidate_revision = None
                    travelled = float(observation.get("travelled_m", 0.0))
                    result = {
                        "error": "premature stop rejected by the configured odometry guard",
                        "travelled_m": travelled,
                        "minimum_travel_m": self.minimum_travel_m,
                        "instruction": "continue navigating and approach the endpoint",
                    }
                elif function_name == "nav.act" and not self._within_budget(
                    arguments, len(action_history)
                ):
                    stop_candidate_revision = None
                    result = {
                        "error": "action batch exceeds the remaining action budget",
                        "remaining_actions": self.max_actions - len(action_history),
                    }
                elif (
                    function_name == "nav.stop"
                    and arguments.get("status") == "completed"
                ):
                    evidence_error = _arrival_evidence_error(arguments)
                    if evidence_error:
                        stop_candidate_revision = None
                        result = {"error": evidence_error}
                    elif not arguments.get("confirm"):
                        stop_candidate_revision = state_revision
                        result = {
                            "error": "arrival candidate recorded; confirmation required",
                            "instruction": (
                                "Recheck every route clause, endpoint identity, and visual "
                                "proximity in the unchanged observation. Call nav_stop with "
                                "confirm=true only if the evidence still holds; otherwise act."
                            ),
                        }
                    elif stop_candidate_revision != state_revision:
                        result = {
                            "error": (
                                "no arrival candidate exists at this state; submit "
                                "confirm=false before the final confirmation"
                            )
                        }
                    else:
                        bus_arguments = _bus_arguments(function_name, arguments)
                        bus_arguments["actor"] = self.name
                        try:
                            result = await context.functions.call(
                                function_name, bus_arguments
                            )
                        except Exception as error:
                            result = {"error": f"{type(error).__name__}: {error}"}
                else:
                    if function_name == "nav.act":
                        stop_candidate_revision = None
                    bus_arguments = _bus_arguments(function_name, arguments)
                    if function_name == "nav.stop":
                        bus_arguments["actor"] = self.name
                    try:
                        result = await context.functions.call(
                            function_name, bus_arguments
                        )
                    except Exception as error:
                        result = {"error": f"{type(error).__name__}: {error}"}

            failed = isinstance(result, Mapping) and "error" in result
            if function_name == "nav.act" and not failed:
                actions = arguments.get("actions", [])
                action_history.extend(str(action) for action in actions)
                state_revision += 1
                if isinstance(result, Mapping) and isinstance(
                    result.get("observation"), Mapping
                ):
                    observation = result["observation"]
            elif function_name == "nav.observe" and isinstance(result, Mapping):
                observation = result

            event = _history_event(
                iteration,
                function_name or call.name,
                arguments,
                result,
                observation,
                _reasoning_summary(response.output),
            )
            all_events.append(event)
            trace = _response_record(iteration, response, calls)
            trace["memory"] = {
                "image_sampling": "uniform_over_trajectory",
                "image_budget": self.image_memory_turns,
                "sampled_history_turns": [
                    turn_history[index][1]["turn"] for index in sampled_history_indices
                ],
                "current_turn": iteration,
                "history_turns": len(turn_history),
                "reasoning_summary": event.get("reasoning_summary"),
            }
            context.output.append_jsonl("model/trace.jsonl", trace)
            context.output.add_artifact("model/trace.jsonl", "application/jsonl")

            if function_name == "nav.stop" and not failed:
                save_history()
                return

            flattened_turn = [
                current_message,
                *response.output,
                _function_output(call.call_id, result),
            ]
            turn_history.append((flattened_turn, event, input_state_revision))
            current_message = _observation_message(
                context.task.instruction,
                observation,
                action_history,
                self.image_detail,
            )

        save_history()
        await context.functions.call(
            "nav.stop",
            status="failed",
            reason="agent iteration limit reached",
            actor=self.name,
        )

    def _within_budget(self, arguments: Mapping[str, Any], used: int) -> bool:
        actions = arguments.get("actions")
        return isinstance(actions, list) and used + len(actions) <= self.max_actions

    def _premature_stop(
        self, arguments: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> bool:
        if arguments.get("status") != "completed":
            return False
        travelled = observation.get("travelled_m", 0.0)
        return float(travelled) < self.minimum_travel_m

    async def _create_response(self, request: dict[str, Any]) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        for attempt in range(self.model_retries + 1):
            try:
                response = self._client.responses.create(**request)
                return await response if inspect.isawaitable(response) else response
            except Exception:
                if attempt >= self.model_retries:
                    raise
                await asyncio.sleep(self.retry_backoff_s * (2**attempt))
        raise AssertionError("unreachable")


def _function_output(call_id: str, result: Any) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(_compact(result), ensure_ascii=True),
    }


def _bus_arguments(function_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if function_name == "nav.act":
        return {"actions": arguments.get("actions")}
    if function_name == "nav.stop":
        return {
            "status": arguments.get("status"),
            "reason": arguments.get("reason"),
            "actor": arguments.get("actor"),
        }
    return {}


def _navigation_memory_error(arguments: Mapping[str, Any]) -> str | None:
    memory = arguments.get("navigation_memory")
    fields = ("current_place", "completed_route", "next_route_step", "decision")
    if not isinstance(memory, Mapping) or any(
        not isinstance(memory.get(field), str) or not memory[field].strip()
        for field in fields
    ):
        return "nav_act requires a non-empty structured navigation_memory ledger"
    return None


def _arrival_evidence_error(arguments: Mapping[str, Any]) -> str | None:
    evidence = arguments.get("arrival_evidence")
    if not isinstance(evidence, Mapping):
        return "completed nav_stop requires structured arrival_evidence"
    required = ("route_complete", "endpoint_visible", "close_enough")
    if any(evidence.get(field) is not True for field in required):
        return (
            "completed nav_stop rejected: route_complete, endpoint_visible, and "
            "close_enough must all be true"
        )
    summary = evidence.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return "completed nav_stop requires a non-empty evidence summary"
    return None


def _memory_input(
    turn_history: list[tuple[list[Any], dict[str, Any], int]],
    current_message: dict[str, Any],
    all_events: list[dict[str, Any]],
    history_checkpoints: int,
    action_history: list[str],
    image_memory_turns: int,
    current_state_revision: int,
) -> tuple[list[Any], list[int]]:
    sampled_indices, continued_index = _sample_history_turns(
        turn_history, current_state_revision, image_memory_turns
    )
    items: list[Any] = []
    if all_events:
        history = _history_data(all_events, history_checkpoints, action_history)
        history["image_context"] = {
            "sampling": "uniform_over_trajectory",
            "sampled_history_turns": [
                turn_history[index][1]["turn"] for index in sampled_indices
            ],
            "current_turn": len(all_events) + 1,
        }
        items.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Navigation history (event checkpoints and the "
                        "model-authored instruction progress ledger). Historical "
                        "image turns below are uniformly sampled and may have gaps:\n"
                        + json.dumps(
                            history,
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
            }
        )
    for index in sampled_indices:
        turn_items, _event, _revision = turn_history[index]
        items.extend(turn_items)
    if continued_index is not None:
        turn_items, _event, _revision = turn_history[continued_index]
        items.extend(_without_images(turn_items))
    items.append(current_message)
    return items, sampled_indices


def _sample_history_turns(
    turn_history: list[tuple[list[Any], dict[str, Any], int]],
    current_state_revision: int,
    image_budget: int,
) -> tuple[list[int], int | None]:
    representative_by_revision: dict[int, int] = {}
    continued_index: int | None = None
    for index, (_items, _event, revision) in enumerate(turn_history):
        if revision < current_state_revision:
            representative_by_revision[revision] = index
        elif revision == current_state_revision:
            continued_index = index
    representatives = list(representative_by_revision.values())
    sampled_positions = _uniform_history_indices(len(representatives), image_budget)
    return [representatives[index] for index in sampled_positions], continued_index


def _without_images(items: list[Any]) -> list[Any]:
    values: list[Any] = []
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("content"), list):
            values.append(item)
            continue
        value = dict(item)
        value["content"] = [
            part
            for part in item["content"]
            if not isinstance(part, Mapping) or part.get("type") != "input_image"
        ]
        values.append(value)
    return values


def _uniform_history_indices(history_size: int, image_budget: int) -> list[int]:
    """Choose history observations evenly between the initial and current states."""
    history_slots = min(history_size, max(0, image_budget - 1))
    if history_slots == 0:
        return []
    if history_slots == history_size:
        return list(range(history_size))
    return [
        (index * history_size + history_slots // 2) // history_slots
        for index in range(history_slots)
    ]


def _history_data(
    events: list[dict[str, Any]],
    history_checkpoints: int,
    action_history: list[str],
) -> dict[str, Any]:
    checkpoints = events[-history_checkpoints:]
    return {
        "total_turns": len(events),
        "omitted_oldest_turns": max(0, len(events) - len(checkpoints)),
        "action_summary": _action_summary(action_history),
        "model_instruction_progress": _model_instruction_progress(events),
        "checkpoints": checkpoints,
    }


def _model_instruction_progress(
    events: list[dict[str, Any]],
) -> dict[str, str] | None:
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        arguments = event.get("arguments")
        evidence = (
            arguments.get("arrival_evidence")
            if isinstance(arguments, Mapping)
            else None
        )
        if (
            event.get("function") == "nav.stop"
            and "error" not in event
            and isinstance(arguments, Mapping)
            and arguments.get("status") == "completed"
            and isinstance(evidence, Mapping)
            and evidence.get("route_complete") is True
            and isinstance(evidence.get("summary"), str)
        ):
            previous = _model_instruction_progress(events[:index]) or {}
            return {
                "current_place": previous.get("current_place", "final observation"),
                "completed_instruction_steps": evidence["summary"],
                "next_instruction_step": "none; model declared route complete",
                "last_decision": str(arguments.get("reason", "stop")),
            }
        arguments = event.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        memory = arguments.get("navigation_memory")
        if not isinstance(memory, Mapping):
            continue
        fields = ("current_place", "completed_route", "next_route_step", "decision")
        if any(not isinstance(memory.get(field), str) for field in fields):
            continue
        return {
            "current_place": memory["current_place"],
            "completed_instruction_steps": memory["completed_route"],
            "next_instruction_step": memory["next_route_step"],
            "last_decision": memory["decision"],
        }
    return None


def _history_event(
    iteration: int,
    function_name: str,
    arguments: Mapping[str, Any],
    result: Any,
    observation: Mapping[str, Any],
    reasoning_summary: str | None,
) -> dict[str, Any]:
    feedback = result.get("actions", []) if isinstance(result, Mapping) else []
    blocked = [
        item.get("action")
        for item in feedback
        if isinstance(item, Mapping) and item.get("blocked")
    ]
    event = {
        "turn": iteration,
        "function": function_name,
        "arguments": _compact(arguments),
        "step": observation.get("step"),
        "travelled_m": observation.get("travelled_m"),
        "pose": _compact(observation.get("pose")),
        "blocked_actions": blocked,
    }
    if reasoning_summary:
        event["reasoning_summary"] = reasoning_summary[:800]
    if isinstance(result, Mapping) and "error" in result:
        event["error"] = str(result["error"])
    return event


def _reasoning_summary(output: list[Any]) -> str | None:
    texts: list[str] = []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for part in getattr(item, "summary", ()) or ():
            text = (
                part.get("text")
                if isinstance(part, Mapping)
                else getattr(part, "text", None)
            )
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n".join(texts) or None


def _action_summary(history: list[str]) -> dict[str, Any]:
    return {
        "total": len(history),
        "move_forward": history.count("move_forward"),
        "turn_left": history.count("turn_left"),
        "turn_right": history.count("turn_right"),
        "recent": history[-12:],
    }


def _model_tools(specs: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tools: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for spec in specs:
        model_name = spec.name.replace(".", "_")
        if model_name in names:
            raise ValueError(f"model tool name collision: {model_name}")
        tool = spec.as_model_tool()
        tool["name"] = model_name
        parameters = dict(tool["parameters"])
        properties = dict(parameters.get("properties", {}))
        required = list(parameters.get("required", []))
        if spec.name == "nav.act":
            properties["navigation_memory"] = _navigation_memory_schema()
            required.append("navigation_memory")
            tool["description"] = (
                f"{spec.description} Include the persistent route ledger used by the "
                "next model turn."
            )
        elif spec.name == "nav.stop":
            properties["status"] = {
                "type": "string",
                "enum": ["completed", "failed", "incomplete"],
            }
            properties["confirm"] = {
                "type": "boolean",
                "description": (
                    "Use false to register an arrival candidate, then true on the "
                    "unchanged state to confirm it."
                ),
            }
            properties["arrival_evidence"] = _arrival_evidence_schema()
            required.extend(["confirm", "arrival_evidence"])
            tool["description"] = (
                f"{spec.description} Completed stops require grounded evidence and "
                "two-stage confirmation."
            )
        parameters["properties"] = properties
        parameters["required"] = list(dict.fromkeys(required))
        tool["parameters"] = parameters
        tools.append(tool)
        names[model_name] = spec.name
    return tools, names


def _navigation_memory_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "current_place": {
                "type": "string",
                "maxLength": 160,
                "description": "Current room or corridor based only on visible evidence.",
            },
            "completed_route": {
                "type": "string",
                "minLength": 1,
                "maxLength": 320,
                "description": (
                    "Your compact summary of ordered instruction clauses already "
                    "verified as complete; use 'none' before any clause is complete."
                ),
            },
            "next_route_step": {
                "type": "string",
                "maxLength": 200,
                "description": "Exactly the next uncompleted clause or landmark.",
            },
            "decision": {
                "type": "string",
                "maxLength": 200,
                "description": "Why this action advances the next route step.",
            },
        },
        "required": [
            "current_place",
            "completed_route",
            "next_route_step",
            "decision",
        ],
        "additionalProperties": False,
    }


def _arrival_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "route_complete": {"type": "boolean"},
            "endpoint_visible": {"type": "boolean"},
            "close_enough": {"type": "boolean"},
            "summary": {
                "type": "string",
                "maxLength": 320,
                "description": (
                    "Model-authored ordered summary of the completed instruction "
                    "clauses and visible endpoint evidence at human standing distance."
                ),
            },
        },
        "required": [
            "route_complete",
            "endpoint_visible",
            "close_enough",
            "summary",
        ],
        "additionalProperties": False,
    }


def _observation_message(
    instruction: str,
    observation: Mapping[str, Any],
    history: list[str],
    image_detail: str,
) -> dict[str, Any]:
    state = {
        "route_instruction": instruction,
        "observation": _compact_observation(observation),
        "action_summary": _action_summary(history),
        "navigation_diagnostics": _navigation_diagnostics(history),
    }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "Current navigation state:\n"
            + json.dumps(state, ensure_ascii=True, separators=(",", ":")),
        }
    ]
    rgb = observation.get("rgb")
    if rgb is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": _jpeg_data_url(rgb),
                "detail": image_detail,
            }
        )
    return {"role": "user", "content": content}


def _navigation_diagnostics(history: list[str]) -> dict[str, Any]:
    trailing_turns: list[str] = []
    for action in reversed(history):
        if action == "move_forward":
            break
        trailing_turns.append(action)
    trailing_turns.reverse()
    direction_changes = sum(
        left != right for left, right in zip(trailing_turns, trailing_turns[1:])
    )
    recent = history[-24:]
    recent_forward = recent.count("move_forward")
    loop_risk = len(trailing_turns) >= 24 or (
        len(trailing_turns) >= 8 and direction_changes > 0
    )
    return {
        "trailing_turn_actions": len(trailing_turns),
        "turn_direction_changes_since_forward": direction_changes,
        "recent_forward_actions": recent_forward,
        "recent_turn_actions": len(recent) - recent_forward,
        "search_loop_risk": loop_risk,
    }


def _compact_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        str(key): _compact(item)
        for key, item in observation.items()
        if key not in {"rgb", "depth"}
    }
    depth = observation.get("depth")
    if depth is not None:
        value["depth_grid_m"] = _depth_grid(depth)
    return value


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
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": list(shape)}
    return str(value)


def _depth_grid(depth: Any) -> list[list[float | None]]:
    import numpy as np

    array = np.asarray(depth, dtype=np.float32).squeeze()
    if array.ndim != 2 or array.size == 0:
        return []
    rows = np.array_split(array, 3, axis=0)
    grid: list[list[float | None]] = []
    for row in rows:
        cells = np.array_split(row, 3, axis=1)
        values: list[float | None] = []
        for cell in cells:
            finite = cell[np.isfinite(cell) & (cell > 0)]
            values.append(round(float(np.median(finite)), 2) if finite.size else None)
        grid.append(values)
    return grid


def _jpeg_data_url(rgb: Any) -> str:
    import numpy as np
    from PIL import Image

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("RGB observation must have shape HxWx3 or HxWx4")
    image = Image.fromarray(array[:, :, :3].astype(np.uint8), "RGB")
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=85)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _response_record(iteration: int, response: Any, calls: list[Any]) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    return {
        "iteration": iteration,
        "response_id": getattr(response, "id", None),
        "usage": _compact(usage),
        "calls": [
            {"name": call.name, "arguments": call.arguments, "call_id": call.call_id}
            for call in calls
        ],
    }


class _AdapterFunctionSpec:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Mapping[str, Any],
        mutates: bool,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = dict(input_schema)
        self.mutates = mutates

    def as_model_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.input_schema),
        }


class _OutputAdapter:
    def __init__(self, context: ModuleContext) -> None:
        self.context = context
        self.metadata: dict[str, Any] = {}
        self.responses = 0
        self.usage: dict[str, int] = {}

    def set_metadata(self, **values: Any) -> None:
        self.metadata.update(values)
        self.context.metadata.update(values)
        self._write_summary()

    def write_json(self, relative_path: str, value: Any) -> None:
        self.context.output.write_json(relative_path, value)

    def append_jsonl(self, relative_path: str, value: Any) -> None:
        path = self.context.output.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_compact(value), sort_keys=True) + "\n")
        if relative_path == "model/trace.jsonl" and isinstance(value, Mapping):
            if value.get("response_id") is not None:
                self.responses += 1
            usage = value.get("usage")
            if isinstance(usage, Mapping):
                for name in ("input_tokens", "output_tokens", "total_tokens"):
                    count = usage.get(name)
                    if isinstance(count, int) and not isinstance(count, bool):
                        self.usage[name] = self.usage.get(name, 0) + count
            self._write_summary()

    def add_artifact(self, relative_path: str, media_type: str) -> None:
        del relative_path, media_type

    def _write_summary(self) -> None:
        self.context.output.write_json(
            "summary.json",
            {
                "schema_version": 1,
                **_compact(self.metadata),
                "model_responses": self.responses,
                "usage": dict(self.usage),
            },
        )


class _TaskAdapter:
    def __init__(self, instruction: Mapping[str, Any]) -> None:
        text = instruction.get("instruction")
        self.instruction = (
            text
            if isinstance(text, str)
            else json.dumps(
                dict(instruction), ensure_ascii=False, separators=(",", ":")
            )
        )


class _FunctionsAdapter:
    def __init__(self, context: ModuleContext, allowed: frozenset[str]) -> None:
        self.context = context
        self.allowed = allowed
        available = {spec.name: spec for spec in context.register.functions()}
        self.aliases: dict[str, str] = {
            "nav.observe": "env.observe",
            "nav.act": "env.step",
            "nav.stop": "env.stop",
        }
        specs: list[_AdapterFunctionSpec] = []
        for name in sorted(allowed):
            spec = available[name]
            if name == "env.observe":
                specs.append(
                    _AdapterFunctionSpec(
                        "nav.observe",
                        "Return the current first-person RGB-D navigation observation and pose.",
                        spec.parameters,
                        False,
                    )
                )
            elif name == "env.step":
                specs.append(
                    _AdapterFunctionSpec(
                        "nav.act",
                        "Execute one to four ordered atomic move or turn actions.",
                        {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "enum": [
                                            "move_forward",
                                            "turn_left",
                                            "turn_right",
                                        ]
                                    },
                                    "minItems": 1,
                                    "maxItems": 4,
                                }
                            },
                            "required": ["actions"],
                            "additionalProperties": False,
                        },
                        True,
                    )
                )
            elif name == "env.stop":
                specs.append(
                    _AdapterFunctionSpec(
                        "nav.stop",
                        "Stop the episode at the current location.",
                        {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "reason": {"type": "string"},
                                "actor": {"type": "string", "minLength": 1},
                            },
                            "required": ["status", "reason", "actor"],
                            "additionalProperties": False,
                        },
                        True,
                    )
                )
            else:
                self.aliases[name] = name
                specs.append(
                    _AdapterFunctionSpec(
                        name,
                        spec.description,
                        spec.parameters,
                        spec.mutates,
                    )
                )
        self.specs = tuple(sorted(specs, key=lambda item: item.name))
        self.action_count = 0
        self.path_length = 0.0
        self.last_feedback: list[dict[str, Any]] = []
        self.observation: dict[str, Any] = {}

    async def call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        **keyword_arguments: Any,
    ) -> Any:
        payload = dict(arguments or {})
        payload.update(keyword_arguments)
        if name == "nav.observe":
            return await self._observe()
        if name == "nav.act":
            return await self._act(payload)
        if name == "nav.stop":
            payload = {
                "status": str(payload.get("status", "failed")),
                "reason": str(payload.get("reason", "")),
                "actor": self.context.name,
            }
            return await self.context.register.acall(
                self.context.name, "env.stop", payload
            )
        target = self.aliases.get(name, name)
        return await self.context.register.acall(self.context.name, target, payload)

    async def _observe(self) -> dict[str, Any]:
        raw = await self.context.register.acall(self.context.name, "env.observe", {})
        self.observation = self._public_observation(raw)
        return self.observation

    async def _act(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        actions = arguments.get("actions")
        if not isinstance(actions, list) or not 1 <= len(actions) <= 4:
            raise ValueError("nav.act requires one to four actions")
        feedback: list[dict[str, Any]] = []
        aliases = {"move_forward": "forward"}
        for action_value in actions:
            action = str(action_value)
            before = _adapter_position(self.observation)
            result = await self.context.register.acall(
                self.context.name,
                "env.step",
                {"action": aliases.get(action, action)},
            )
            after = _adapter_position(result)
            translation = (
                math.dist(before, after)
                if before is not None
                and after is not None
                and len(before) == len(after)
                else None
            )
            if translation is not None:
                self.path_length += translation
            self.action_count += 1
            item = {
                "action": action,
                "translation_m": (
                    round(translation, 4) if translation is not None else None
                ),
                "blocked": action == "move_forward"
                and translation is not None
                and translation < 0.05,
            }
            feedback.append(item)
            if isinstance(result, Mapping) and result.get("terminal"):
                break
            if isinstance(result, Mapping):
                self.observation["pose"] = result.get(
                    "pose", self.observation.get("pose")
                )
        self.last_feedback = feedback
        try:
            observation = await self._observe()
        except Exception:
            observation = self.observation
        return {"actions": feedback, "observation": observation}

    def _public_observation(self, raw: Any) -> dict[str, Any]:
        value = dict(raw) if isinstance(raw, Mapping) else {}
        channels = value.get("channels")
        channels = channels if isinstance(channels, Mapping) else {}
        pose = value.get("pose")
        gps = channels.get("gps")
        compass = channels.get("compass")
        if isinstance(pose, Mapping):
            pose = {
                "position": _adapter_sequence(pose.get("position")),
                "gps": _compact(pose.get("gps", gps)),
                "heading_degrees": pose.get(
                    "heading_degrees", _heading_degrees(compass)
                ),
            }
        else:
            position = _adapter_sequence(pose) or _adapter_sequence(gps)
            pose = {
                "position": position,
                "gps": _compact(gps),
                "heading_degrees": _heading_degrees(compass),
            }
        return {
            "observation_id": value.get("observation_id"),
            "time_unix": time.time(),
            "step": self.action_count,
            "travelled_m": round(self.path_length, 3),
            "motion_scale": {
                "move_forward_m": 0.25,
                "turn_degrees": 15,
            },
            "rgb": channels.get("rgb"),
            "depth": channels.get("depth"),
            "pose": pose,
            "last_actions": list(self.last_feedback),
        }


class _FabricContext:
    def __init__(self, context: ModuleContext, allowed: frozenset[str]) -> None:
        self.output = _OutputAdapter(context)
        self.functions = _FunctionsAdapter(context, allowed)
        self.task = _TaskAdapter(context.episode.instruction)
        self.cancelled = context.cancelled


def _adapter_sequence(value: Any) -> list[float] | None:
    tolist = getattr(value, "tolist", None)
    value = tolist() if callable(tolist) else value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _adapter_position(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, Mapping):
        return None
    pose = value.get("pose")
    if isinstance(pose, Mapping):
        position = pose.get("position")
        if isinstance(position, Mapping):
            position = [
                position.get("x"),
                position.get("y", 0.0),
                position.get("z", 0.0),
            ]
    else:
        position = pose
    values = _adapter_sequence(position)
    return tuple(values) if values is not None else None


def _heading_degrees(compass: Any) -> float | None:
    values = _adapter_sequence(compass)
    if values is None or not values:
        return None
    degrees = math.degrees(values[0]) % 360.0
    return 0.0 if math.isclose(degrees, 360.0, abs_tol=0.01) else round(degrees, 2)

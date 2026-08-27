from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal

from harness.contracts import NavigationStack
from harness.errors import HarnessError
from harness.output import EpisodeOutput, ModuleOutput, NULL_MODULE_OUTPUT
from harness.requirements import check_navigation_requirements
from harness.tool_bus import JsonObject, Tool, ToolBus, ToolClient, ToolEvent
from schemas import EnvironmentTerminal, NavTask


@dataclass(frozen=True, slots=True)
class Terminal:
    status: str
    reason: str
    actor: str


class _TerminalSignal:
    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.value: Terminal | None = None
        self._lock = asyncio.Lock()

    async def claim(self, status: str, reason: str, actor: str) -> tuple[Terminal, bool]:
        async with self._lock:
            claimed = self.value is None
            if self.value is None:
                self.value = Terminal(status, reason, actor)
            return self.value, claimed

    def announce(self) -> None:
        self.event.set()

    async def set(self, status: str, reason: str, actor: str) -> Terminal:
        value, claimed = await self.claim(status, reason, actor)
        if claimed:
            self.announce()
        else:
            await self.event.wait()
        return value


class NavTools:
    def __init__(self, client: ToolClient) -> None:
        self._client = client

    async def observe(self) -> JsonObject:
        return await self._client.call("nav.observe")

    async def move_discrete(self, action: str) -> JsonObject:
        return await self._client.call("nav.move.discrete", action=action)

    async def stop(self, status: str, reason: str = "") -> JsonObject:
        return await self._client.call("nav.stop", status=status, reason=reason)

    async def finish_goal(self, status: str, reason: str = "") -> JsonObject:
        return await self._client.call("nav.goal.finish", status=status, reason=reason)


class VLNTools:
    def __init__(self, client: ToolClient) -> None:
        self._client = client

    async def navigate_task(self, instruction: str) -> JsonObject:
        return await self._client.call("vln.navigate.task", instruction=instruction)

    async def navigate_local(self, instruction: str) -> JsonObject:
        return await self._client.call("vln.navigate.local", instruction=instruction)


class SpatialTools:
    def __init__(self, client: ToolClient) -> None:
        self._client = client

    async def search(
        self,
        query: str = "",
        *,
        frame: str | None = None,
        near_pose: list[float] | None = None,
        top_k: int = 5,
    ) -> list[JsonObject]:
        arguments: JsonObject = {"query": query, "top_k": top_k}
        if frame is not None:
            arguments["frame"] = frame
        if near_pose is not None:
            arguments["near_pose"] = near_pose
        result = await self._client.call("spatial.search", arguments)
        return result["items"]

    async def remember(
        self, text: str, frame: str, pose: list[float] | None = None
    ) -> JsonObject:
        arguments: JsonObject = {"text": text, "frame": frame}
        if pose is not None:
            arguments["pose"] = pose
        return await self._client.call("spatial.remember", arguments)


@dataclass(frozen=True, slots=True)
class NavContext:
    task: NavTask
    execution_id: str
    tools: ToolClient
    cancelled: asyncio.Event
    output: ModuleOutput = NULL_MODULE_OUTPUT

    @property
    def nav(self) -> NavTools:
        return NavTools(self.tools)

    @property
    def vln(self) -> VLNTools:
        return VLNTools(self.tools)

    @property
    def spatial(self) -> SpatialTools:
        return SpatialTools(self.tools)


@dataclass(frozen=True, slots=True)
class NavigationResult:
    execution_id: str
    task_id: str
    terminal: Terminal
    environment: JsonObject
    audit: tuple[ToolEvent, ...]
    cleanup_errors: tuple[str, ...] = ()


class NavigationHarness:
    """Owns one task lifecycle; all navigation decisions remain in the agent."""

    def __init__(self, timeout_s: float = 300.0, shutdown_timeout_s: float = 10.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        self.timeout_s = timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s

    async def run_task(
        self,
        task: NavTask,
        stack: NavigationStack,
        *,
        output: EpisodeOutput | None = None,
    ) -> NavigationResult:
        execution_id = uuid.uuid4().hex
        environment_output = (
            output.module("environment") if output else NULL_MODULE_OUTPUT
        )
        memory_output = output.module("memory") if output else NULL_MODULE_OUTPUT
        vln_output = output.module("vln") if output else NULL_MODULE_OUTPUT
        agent_output = output.module("agent") if output else NULL_MODULE_OUTPUT
        bus = ToolBus()
        terminal = _TerminalSignal()
        cancelled = asyncio.Event()
        cleanup_errors: list[str] = []
        started: list[Literal["environment", "memory", "vln"]] = []
        environment_stop_lock = asyncio.Lock()
        environment_stopped = False

        async def run_cleanup(label: str, operation: Any) -> None:
            try:
                await asyncio.wait_for(operation, timeout=self.shutdown_timeout_s)
            except asyncio.TimeoutError:
                cleanup_errors.append(f"{label}: cleanup timed out")
            except asyncio.CancelledError:
                cleanup_errors.append(f"{label}: cleanup cancelled")
            except Exception as error:
                cleanup_errors.append(f"{label}: {type(error).__name__}: {error}")

        async def stop_environment(reason: str) -> None:
            nonlocal environment_stopped
            if "environment" not in started:
                return
            async with environment_stop_lock:
                if environment_stopped:
                    return
                environment_stopped = True
                await run_cleanup("environment", stack.environment.stop(reason))

        async def stop_tool(actor: str, arguments: JsonObject) -> JsonObject:
            value, claimed = await terminal.claim(
                arguments["status"], arguments.get("reason", ""), actor
            )
            if claimed:
                bus.close_writes()
                await stop_environment(value.reason)
                terminal.announce()
            else:
                await terminal.event.wait()
            return {"status": value.status, "reason": value.reason}

        bus.register(
            (
                Tool(
                    name="nav.stop",
                    description="Finish the complete navigation task.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "minLength": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["status", "reason"],
                        "additionalProperties": False,
                    },
                    handler=stop_tool,
                    writes=True,
                ),
            ),
            owner="harness",
        )

        agent_task: asyncio.Task[None] | None = None
        failure_task: asyncio.Task[EnvironmentTerminal] | None = None
        terminal_task: asyncio.Task[bool] | None = None
        propagate_cancel = False
        try:
            if stack.vln is not None and stack.vln.requirements:
                check_navigation_requirements(
                    type(stack.vln).__name__,
                    stack.vln.requirements,
                    stack.environment.profile,
                )

            started.append("environment")
            environment_tools = (
                await stack.environment.start(task, environment_output)
                if output is not None
                else await stack.environment.start(task)
            )
            bus.register(
                environment_tools,
                owner=f"environment {type(stack.environment).__name__}",
            )

            if stack.memory is not None:
                bus.require(type(stack.memory).__name__, stack.memory.required_tools)
                memory_tools = bus.client("memory", stack.memory.required_tools)
                started.append("memory")
                memory_bindings = (
                    await stack.memory.start(task, memory_tools, memory_output)
                    if output is not None
                    else await stack.memory.start(task, memory_tools)
                )
                bus.register(
                    memory_bindings,
                    owner=f"memory {type(stack.memory).__name__}",
                )

            if stack.vln is not None:
                bus.require(type(stack.vln).__name__, stack.vln.required_tools)
                vln_tools = bus.client("vln", stack.vln.required_tools)
                started.append("vln")
                vln_bindings = (
                    await stack.vln.start(task, vln_tools, vln_output)
                    if output is not None
                    else await stack.vln.start(task, vln_tools)
                )
                bus.register(
                    vln_bindings,
                    owner=f"vln {type(stack.vln).__name__}",
                )

            agent_tools = frozenset(stack.agent.required_tools) | {"nav.stop"}
            bus.require(type(stack.agent).__name__, agent_tools)
            context = NavContext(
                task=task,
                execution_id=execution_id,
                tools=bus.client("agent", agent_tools),
                cancelled=cancelled,
                output=agent_output,
            )
            agent_task = asyncio.create_task(stack.agent.run(context), name="agent")
            failure_task = asyncio.create_task(
                stack.environment.wait_terminal(), name="environment-terminal"
            )
            terminal_task = asyncio.create_task(terminal.event.wait(), name="terminal")

            done, _ = await asyncio.wait(
                (agent_task, failure_task, terminal_task),
                timeout=self.timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await terminal.set("timeout", "navigation task timed out", "harness")
            elif terminal_task in done:
                pass
            elif failure_task in done:
                error = failure_task.exception()
                if error is not None:
                    await terminal.set(
                        "failed", f"environment monitor failed: {error}", "harness"
                    )
                else:
                    native = failure_task.result()
                    status = "environment_terminal" if native.kind == "completed" else "failed"
                    await terminal.set(status, native.reason, "environment")
            else:
                error = agent_task.exception()
                if error is None:
                    reason = "agent returned without calling nav.stop"
                else:
                    reason = f"agent failed: {type(error).__name__}: {error}"
                await terminal.set("failed", reason, "harness")
        except asyncio.CancelledError:
            propagate_cancel = True
            await terminal.set("cancelled", "execution cancelled", "harness")
        except Exception as error:
            await terminal.set(
                "failed", f"startup failed: {type(error).__name__}: {error}", "harness"
            )
        finally:
            bus.close_writes()
            cancelled.set()
            reason = terminal.value.reason if terminal.value else "lifecycle ended"

            # Fence native motion before stopping producers of motion commands.
            await stop_environment(reason)
            if "vln" in started and stack.vln is not None:
                await run_cleanup("vln", stack.vln.stop(reason))

            for handle in (agent_task, failure_task, terminal_task):
                if handle is not None and not handle.done():
                    handle.cancel()
            handles = [
                handle
                for handle in (agent_task, failure_task, terminal_task)
                if handle is not None
            ]
            if handles:
                await asyncio.gather(*handles, return_exceptions=True)

            await run_cleanup("tool writes", bus.drain_writes())
            if "memory" in started and stack.memory is not None:
                await run_cleanup("memory", stack.memory.stop(reason))

        if propagate_cancel:
            raise asyncio.CancelledError()

        final_terminal = terminal.value or Terminal(
            "failed", "task ended without terminal state", "harness"
        )
        environment_result: JsonObject = {}
        if "environment" in started:
            try:
                environment_result = stack.environment.result()
                environment_output.record({"result": environment_result})
            except Exception as error:
                cleanup_errors.append(f"result: {type(error).__name__}: {error}")
        if output is not None:
            for event in bus.audit:
                output.event(asdict(event))
        return NavigationResult(
            execution_id=execution_id,
            task_id=task.task_id,
            terminal=final_terminal,
            environment=environment_result,
            audit=bus.audit,
            cleanup_errors=tuple(cleanup_errors),
        )

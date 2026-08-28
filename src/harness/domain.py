from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from harness.components import Component, ComponentContext, DomainComponents, Metric
from harness.errors import ContractError
from harness.functions import FunctionBus
from harness.output import ComponentManifest, DomainOutput
from schemas import JsonObject, NavigationTask, Terminal


@dataclass(frozen=True, slots=True)
class DomainResult:
    domain_id: str
    task_id: str
    terminal: Terminal
    environment: JsonObject
    metrics: dict[str, float]
    components: dict[str, ComponentManifest]
    cleanup_errors: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> JsonObject:
        return {
            "schema_version": 1,
            "domain_id": self.domain_id,
            "task_id": self.task_id,
            "terminal": self.terminal.as_dict(),
            "environment": dict(self.environment),
            "metrics": dict(self.metrics),
            "components": {
                name: manifest.as_dict() for name, manifest in self.components.items()
            },
            "cleanup_errors": list(self.cleanup_errors),
        }


class DomainRuntime:
    """Run one complete environment lifetime without driving navigation steps."""

    def __init__(self, timeout_s: float = 300.0, shutdown_timeout_s: float = 10.0) -> None:
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be a positive finite number")
        if shutdown_timeout_s <= 0 or not math.isfinite(shutdown_timeout_s):
            raise ValueError("shutdown_timeout_s must be a positive finite number")
        self.timeout_s = timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s

    async def run(
        self,
        task: NavigationTask,
        components: DomainComponents,
        *,
        output_root: str | None = None,
        domain_id: str | None = None,
    ) -> DomainResult:
        identifier = domain_id or uuid.uuid4().hex
        self._validate_components(components)
        output = DomainOutput(output_root, identifier)
        bus = FunctionBus()
        cancelled = asyncio.Event()
        contexts: dict[str, ComponentContext] = {}

        for component in components.all:
            bus.register(component.name, tuple(component.functions()))
        if "nav.stop" not in bus.names:
            raise ContractError("environment must provide nav.stop")
        if bus.owner("nav.stop") != components.environment.name:
            raise ContractError("nav.stop must be provided by the environment")
        bus.seal()

        for component in components.all:
            required = frozenset(component.required_functions)
            optional = frozenset(component.optional_functions) & bus.names
            bus.require(component.name, required)
            contexts[component.name] = ComponentContext(
                domain_id=identifier,
                task=task,
                functions=bus.client(component.name, required | optional),
                output=output.component(component.name),
                cancelled=cancelled,
            )

        system = bus.client("domain", None)
        started: list[Component] = []
        cleanup_errors: list[str] = []
        agent_task: asyncio.Task[None] | None = None
        terminal_task: asyncio.Task[Terminal] | None = None
        terminal: Terminal | None = None
        environment_result: JsonObject = {}
        metrics: dict[str, float] = {}
        cancellation: asyncio.CancelledError | None = None

        try:
            await self._start(components.environment, contexts, started)
            for component in (*components.services, *components.metrics, components.agent):
                await self._start(component, contexts, started)

            agent_task = asyncio.create_task(
                components.agent.run(contexts[components.agent.name]),
                name=f"{identifier}:agent",
            )
            terminal_task = asyncio.create_task(
                components.environment.wait_terminal(),
                name=f"{identifier}:terminal",
            )
            terminal = await self._wait_for_terminal(
                agent_task, terminal_task, system
            )
        except asyncio.CancelledError as error:
            cancellation = error
            terminal = await self._request_stop(
                system, "cancelled", "Domain execution cancelled", "domain"
            )
        except Exception as error:
            terminal = await self._request_stop(
                system,
                "failed",
                f"Domain execution failed: {type(error).__name__}: {error}",
                "domain",
            )
        finally:
            cancelled.set()
            bus.close_writes()
            await self._settle_task(agent_task, cleanup_errors, "agent")
            await self._settle_task(terminal_task, cleanup_errors, "environment terminal")
            try:
                await bus.wait_for_mutations(self.shutdown_timeout_s)
            except BaseException as error:
                cleanup_errors.append(_error("function mutations", error))

            if terminal is None:
                terminal = Terminal("failed", "environment did not report terminal", "domain")
            try:
                environment_result = dict(await components.environment.result())
            except BaseException as error:
                cleanup_errors.append(_error("environment result", error))

            for metric in components.metrics:
                if metric not in started:
                    continue
                try:
                    values = await metric.evaluate(terminal, environment_result)
                    self._merge_metrics(metric, values, metrics)
                except BaseException as error:
                    cleanup_errors.append(_error(f"metric {metric.name}", error))

            for component in reversed(started):
                try:
                    await asyncio.wait_for(
                        component.close(terminal.reason),
                        timeout=self.shutdown_timeout_s,
                    )
                except BaseException as error:
                    cleanup_errors.append(_error(f"component {component.name}", error))
            try:
                await bus.close(self.shutdown_timeout_s)
            except BaseException as error:
                cleanup_errors.append(_error("function bus", error))

        manifests = {
            component.name: output.component(component.name).manifest()
            for component in components.all
        }
        result = DomainResult(
            domain_id=identifier,
            task_id=task.task_id,
            terminal=terminal,
            environment=environment_result,
            metrics=metrics,
            components=manifests,
            cleanup_errors=tuple(cleanup_errors),
        )
        try:
            output.finish(result, [record.as_dict() for record in bus.records])
        except BaseException as error:
            cleanup_errors.append(_error("domain output", error))
            result = DomainResult(
                domain_id=result.domain_id,
                task_id=result.task_id,
                terminal=result.terminal,
                environment=result.environment,
                metrics=result.metrics,
                components=result.components,
                cleanup_errors=tuple(cleanup_errors),
            )

        if cancellation is not None:
            raise cancellation
        return result

    async def _wait_for_terminal(
        self,
        agent_task: asyncio.Task[None],
        terminal_task: asyncio.Task[Terminal],
        system: Any,
    ) -> Terminal:
        done, _ = await asyncio.wait(
            (agent_task, terminal_task),
            timeout=self.timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            return await self._request_stop(
                system, "timeout", "navigation Domain timed out", "domain"
            )
        if terminal_task in done:
            return terminal_task.result()

        error = agent_task.exception()
        if error is not None:
            return await self._request_stop(
                system,
                "failed",
                f"Agent failed: {type(error).__name__}: {error}",
                "agent",
            )
        if terminal_task.done():
            return terminal_task.result()
        return await self._request_stop(
            system,
            "failed",
            "Agent returned without stopping the environment",
            "domain",
        )

    async def _request_stop(
        self,
        system: Any,
        status: str,
        reason: str,
        actor: str,
    ) -> Terminal:
        try:
            value = await system.call(
                "nav.stop", status=status, reason=reason, actor=actor
            )
            if isinstance(value, Terminal):
                return value
            if isinstance(value, Mapping):
                return Terminal(
                    status=str(value.get("status", status)),
                    reason=str(value.get("reason", reason)),
                    actor=str(value.get("actor", actor)),
                )
        except BaseException as error:
            return Terminal(
                "failed",
                f"{reason}; stop failed: {type(error).__name__}: {error}",
                "domain",
            )
        return Terminal(status, reason, actor)

    async def _settle_task(
        self,
        task: asyncio.Task[Any] | None,
        errors: list[str],
        label: str,
    ) -> None:
        if task is None or task.done():
            if task is not None:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as error:
                    if label != "agent":
                        errors.append(_error(label, error))
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), self.shutdown_timeout_s)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            errors.append(f"{label}: shutdown timed out")
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except BaseException as error:
            if label != "agent":
                errors.append(_error(label, error))

    async def _start(
        self,
        component: Component,
        contexts: Mapping[str, ComponentContext],
        started: list[Component],
    ) -> None:
        await component.start(contexts[component.name])
        started.append(component)

    @staticmethod
    def _validate_components(components: DomainComponents) -> None:
        names: set[str] = set()
        for component in components.all:
            if not component.name or not component.name.strip():
                raise ContractError(
                    f"component {type(component).__name__} must declare a non-empty name"
                )
            if component.name in names:
                raise ContractError(f"duplicate component name {component.name!r}")
            names.add(component.name)
            if not isinstance(component.required_functions, frozenset):
                raise ContractError(
                    f"component {component.name!r} required_functions must be a frozenset"
                )
            if not isinstance(component.optional_functions, frozenset):
                raise ContractError(
                    f"component {component.name!r} optional_functions must be a frozenset"
                )

    @staticmethod
    def _merge_metrics(
        metric: Metric,
        values: Mapping[str, float],
        target: dict[str, float],
    ) -> None:
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ContractError(f"metric {metric.name!r} returned an invalid name")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(
                    f"metric {metric.name!r} value {name!r} must be numeric"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ContractError(
                    f"metric {metric.name!r} value {name!r} must be finite"
                )
            if name in target:
                raise ContractError(f"duplicate metric name {name!r}")
            target[name] = number


def _error(label: str, error: BaseException) -> str:
    return f"{label}: {type(error).__name__}: {error}"

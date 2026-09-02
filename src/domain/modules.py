from __future__ import annotations

import importlib
import asyncio
import inspect
import math
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from domain.contracts import ModuleSpec, NavigationEpisode, Terminal
from domain.errors import HarnessError
from domain.register import DomainRegister
from domain.workspace import ModuleWorkspace, Workspace


@dataclass(slots=True)
class ModuleContext:
    domain_id: str
    name: str
    episode: NavigationEpisode
    register: DomainRegister
    workspace: Workspace
    output: ModuleWorkspace
    cancelled: threading.Event
    metadata: dict[str, Any] = field(default_factory=dict)


class Module(ABC):
    """One configurable unit running in its own Domain thread."""

    def _bind(self, context: ModuleContext) -> None:
        self.context = context
        self._requests: queue.Queue[_Request] = queue.Queue()
        self._service_started = threading.Event()
        self._service_stopped = threading.Event()
        self._owner_thread: int | None = None
        self._run_error: BaseException | None = None

    def mount(self) -> None:
        """Register functions and references before module threads start."""

    def run(self) -> None:
        self.serve_until(self.context.cancelled.is_set)

    def close(self) -> None:
        """Release resources after all module threads are asked to stop."""

    def expose(
        self,
        name: str,
        handler: Any,
        *,
        description: str,
        parameters: Mapping[str, Any] | None = None,
        mutates: bool = False,
        serial_key: str | None = None,
    ) -> None:
        """Register a synchronous function whose handler runs on this module's thread."""
        self.context.register.register_function(
            self.context.name,
            name,
            self._dispatch(handler),
            description=description,
            parameters=parameters,
            mutates=mutates,
            serial_key=serial_key,
        )

    def serve_until(self, stopped: Any) -> None:
        self._owner_thread = threading.get_ident()
        self._service_started.set()
        try:
            while not stopped():
                try:
                    request = self._requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    request.result = _resolve(request.handler(**request.arguments))
                except BaseException as error:
                    request.error = error
                finally:
                    request.completed.set()
        finally:
            self._service_stopped.set()
            service_error = HarnessError(
                f"module {self.context.name!r} function service stopped"
            )
            while True:
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    break
                request.error = service_error
                request.completed.set()

    def _dispatch(self, handler: Any) -> Any:
        def call(**arguments: Any) -> Any:
            if threading.get_ident() == self._owner_thread:
                return _resolve(handler(**arguments))
            if self._run_error is not None:
                raise HarnessError(
                    f"module {self.context.name!r} failed: "
                    f"{type(self._run_error).__name__}: {self._run_error}"
                ) from self._run_error
            if self._service_stopped.is_set():
                raise HarnessError(f"module {self.context.name!r} function service is stopped")
            request = _Request(handler, arguments)
            self._requests.put(request)
            while not request.completed.wait(0.05):
                if self._run_error is not None or self._service_stopped.is_set():
                    raise HarnessError(
                        f"module {self.context.name!r} stopped before handling {handler.__name__}"
                    ) from self._run_error
            if request.error is not None:
                raise request.error
            return request.result

        return call

    def _fail(self, error: BaseException) -> None:
        self._run_error = error
        self._service_stopped.set()


@dataclass(slots=True)
class _Request:
    handler: Any
    arguments: dict[str, Any]
    completed: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class EnvironmentModule(Module, ABC):
    def __init__(self) -> None:
        self._terminal: Terminal | None = None
        self._terminal_event = threading.Event()
        self._ready = threading.Event()
        self._terminal_lock = threading.RLock()

    def _bind(self, context: ModuleContext) -> None:
        super()._bind(context)
        self.expose(
            "env.stop",
            self._stop,
            description="End this navigation environment and freeze its final state.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "reason": {"type": "string"},
                    "actor": {"type": "string"},
                },
                "required": ["status"],
                "additionalProperties": False,
            },
            mutates=True,
            serial_key="env.state",
        )
        context.register.register_reference(
            context.name,
            "env.ready",
            self._ready.is_set,
            description="Whether the native environment is ready for calls.",
        )
        context.register.register_reference(
            context.name,
            "env.terminal",
            lambda: self._terminal.as_dict() if self._terminal else None,
            description="The immutable terminal state, or null while running.",
        )

    def run(self) -> None:
        primary: BaseException | None = None
        try:
            self.start()
            self._ready.set()
            self.serve_until(self._terminal_event.is_set)
        except BaseException as error:
            primary = error
            self.finish("failed", f"environment start failed: {type(error).__name__}: {error}", "env")
            raise
        finally:
            self._ready.set()
            try:
                self.close()
            except BaseException:
                if primary is None:
                    raise

    def start(self) -> None:
        """Initialize the native simulator or robot session."""

    def wait_ready(self, timeout_s: float | None = None) -> bool:
        return self._ready.wait(timeout_s)

    def wait_terminal(self, timeout_s: float | None = None) -> Terminal | None:
        return self._terminal if self._terminal_event.wait(timeout_s) else None

    def finish(self, status: str, reason: str, actor: str = "env") -> Terminal:
        with self._terminal_lock:
            if self._terminal is None:
                self._terminal = Terminal(status, reason, actor)
                self._terminal_event.set()
            return self._terminal

    def _stop(
        self,
        status: str,
        reason: str = "",
        actor: str = "module",
    ) -> dict[str, Any]:
        with self._terminal_lock:
            if self._terminal is not None:
                return self._terminal.as_dict()
            try:
                self.stop(reason)
            except BaseException as error:
                self.finish(
                    "failed", f"environment stop failed: {type(error).__name__}: {error}", actor
                )
                raise
            return self.finish(status, reason, actor).as_dict()

    def stop(self, reason: str) -> None:
        """Stop native motion. This method must be idempotent."""

    @abstractmethod
    def result(self) -> Mapping[str, Any]:
        raise NotImplementedError


class MetricModule(Module, ABC):
    @abstractmethod
    def evaluate(
        self, terminal: Terminal, environment: Mapping[str, Any]
    ) -> Mapping[str, float]:
        raise NotImplementedError


class ModuleRegistry:
    """Build and own all module instances for one Domain."""

    def __init__(self, specs: tuple[ModuleSpec, ...]) -> None:
        self.specs = specs
        self.modules: dict[str, Module] = {}
        self.contexts: dict[str, ModuleContext] = {}

    def build(
        self,
        *,
        domain_id: str,
        episode: NavigationEpisode,
        register: DomainRegister,
        workspace: Workspace,
        cancelled: threading.Event,
    ) -> None:
        for spec in self.specs:
            module = _instantiate(spec)
            context = ModuleContext(
                domain_id,
                spec.name,
                episode,
                register,
                workspace,
                workspace.module(spec.name),
                cancelled,
            )
            module._bind(context)
            self.modules[spec.name] = module
            self.contexts[spec.name] = context
        if not isinstance(self.modules.get("env"), EnvironmentModule):
            raise HarnessError("configured env must be an EnvironmentModule")
        if not isinstance(self.modules.get("metric"), MetricModule):
            raise HarnessError("configured metric must be a MetricModule")

    @property
    def environment(self) -> EnvironmentModule:
        return self.modules["env"]  # type: ignore[return-value]

    @property
    def metric(self) -> MetricModule:
        return self.modules["metric"]  # type: ignore[return-value]

    def mount(self) -> None:
        for module in self.modules.values():
            module.mount()

    def close(self) -> tuple[str, ...]:
        errors: list[str] = []
        for name, module in reversed(tuple(self.modules.items())):
            try:
                module.close()
            except BaseException as error:
                errors.append(f"module {name} close: {type(error).__name__}: {error}")
        return tuple(errors)

    def manifest(self) -> dict[str, Mapping[str, Any]]:
        values: dict[str, Mapping[str, Any]] = {}
        by_name = {spec.name: spec for spec in self.specs}
        for name, context in self.contexts.items():
            files = sorted(
                str(path.relative_to(context.output.root))
                for path in context.output.root.rglob("*")
                if path.is_file()
            )
            values[name] = {
                "factory": by_name[name].factory,
                "metadata": dict(context.metadata),
                "files": files,
            }
        return values


def validate_metrics(values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise HarnessError("metric evaluate() must return a mapping")
    result: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise HarnessError("metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HarnessError(f"metric {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise HarnessError(f"metric {name!r} must be finite")
        result[name] = numeric
    return result


def _instantiate(spec: ModuleSpec) -> Module:
    module_name, attribute = spec.factory.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        value = factory(**dict(spec.params))
    except BaseException as error:
        raise HarnessError(
            f"module {spec.name!r} factory {spec.factory!r} failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not isinstance(value, Module):
        raise HarnessError(
            f"module {spec.name!r} factory returned {type(value).__name__}, expected Module"
        )
    return value


def _resolve(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    return asyncio.run(_await_value(value))


async def _await_value(value: Any) -> Any:
    return await value

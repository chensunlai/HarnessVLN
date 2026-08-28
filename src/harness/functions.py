from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from itertools import count
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError, ValidationError  # type: ignore[import-untyped]

from harness.errors import (
    ContractError,
    DomainClosedError,
    FunctionNotFoundError,
    FunctionPermissionError,
    FunctionValidationError,
)
from schemas import JsonObject


_FUNCTION_NAME = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_OBJECT_SCHEMA: JsonObject = {"type": "object"}


@dataclass(frozen=True, slots=True)
class CallContext:
    call_id: int
    actor: str
    function: str


FunctionHandler = Callable[[CallContext, JsonObject], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class Function:
    name: str
    description: str
    handler: FunctionHandler
    input_schema: JsonObject | None = None
    output_schema: JsonObject | None = None
    mutates: bool = False
    serial_key: str | None = None

    def __post_init__(self) -> None:
        if not _FUNCTION_NAME.fullmatch(self.name):
            raise ContractError(
                f"invalid function name {self.name!r}; use a dotted lowercase namespace"
            )
        if not self.description.strip():
            raise ContractError(f"function {self.name!r} requires a description")
        if not callable(self.handler):
            raise ContractError(f"function {self.name!r} handler must be callable")
        if self.serial_key is not None and not self.serial_key.strip():
            raise ContractError(f"function {self.name!r} serial_key must not be empty")
        for direction, schema in (
            ("input", self.input_schema),
            ("output", self.output_schema),
        ):
            if schema is None:
                continue
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as error:
                raise ContractError(
                    f"function {self.name!r} has an invalid {direction} schema: {error.message}"
                ) from error

    @property
    def spec(self) -> "FunctionSpec":
        return FunctionSpec(
            name=self.name,
            description=self.description,
            input_schema=dict(self.input_schema or _OBJECT_SCHEMA),
            mutates=self.mutates,
        )


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    description: str
    input_schema: JsonObject
    mutates: bool

    def as_model_tool(self) -> JsonObject:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class CallRecord:
    call_id: int
    started_at: float
    duration_s: float
    actor: str
    function: str
    arguments: JsonObject
    outcome: str
    result: Any = None
    error: str | None = None

    def as_dict(self) -> JsonObject:
        return asdict(self)


class FunctionBus:
    """A Domain-local function directory with capability and mutation boundaries."""

    def __init__(self) -> None:
        self._functions: dict[str, Function] = {}
        self._owners: dict[str, str] = {}
        self._serial_locks: dict[str, asyncio.Lock] = {}
        self._ids = count(1)
        self._records: list[CallRecord] = []
        self._sealed = False
        self._closed = False
        self._writes_open = True
        self._active_count = 0
        self._active_mutation_count = 0
        self._idle = asyncio.Event()
        self._mutations_idle = asyncio.Event()
        self._idle.set()
        self._mutations_idle.set()

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(sorted(self._records, key=lambda record: record.call_id))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._functions)

    def owner(self, name: str) -> str:
        try:
            return self._owners[name]
        except KeyError as error:
            raise FunctionNotFoundError(f"unknown function {name!r}") from error

    def register(self, owner: str, functions: Sequence[Function]) -> None:
        if self._sealed:
            raise ContractError("functions cannot be registered after the bus is sealed")
        if not owner.strip():
            raise ContractError("function owner must not be empty")
        for function in functions:
            if function.name in self._functions:
                previous = self._owners[function.name]
                raise ContractError(
                    f"function {function.name!r} is provided by both {previous!r} and {owner!r}"
                )
            self._functions[function.name] = function
            self._owners[function.name] = owner
            if function.serial_key is not None:
                self._serial_locks.setdefault(function.serial_key, asyncio.Lock())

    def seal(self) -> None:
        self._sealed = True

    def require(self, component: str, names: Iterable[str]) -> None:
        missing = sorted(set(names) - self.names)
        if missing:
            raise ContractError(
                f"component {component!r} requires unavailable functions: {', '.join(missing)}"
            )

    def client(self, actor: str, allowed: Iterable[str] | None) -> "FunctionClient":
        if not actor.strip():
            raise ContractError("function actor must not be empty")
        capability_set = None if allowed is None else frozenset(allowed)
        if capability_set is not None:
            self.require(actor, capability_set)
        return FunctionClient(self, actor, capability_set)

    def specs(self, allowed: frozenset[str] | None) -> tuple[FunctionSpec, ...]:
        names = self.names if allowed is None else allowed
        return tuple(self._functions[name].spec for name in sorted(names))

    def close_writes(self) -> None:
        self._writes_open = False

    async def wait_for_mutations(self, timeout_s: float | None = None) -> None:
        await _wait(self._mutations_idle, timeout_s)

    async def close(self, timeout_s: float | None = None) -> None:
        self.close_writes()
        self._closed = True
        await _wait(self._idle, timeout_s)

    async def call(
        self,
        actor: str,
        name: str,
        arguments: Mapping[str, Any],
        allowed: frozenset[str] | None,
    ) -> Any:
        if not self._sealed:
            raise ContractError("the function bus must be sealed before calls are accepted")
        if self._closed:
            raise DomainClosedError("the Domain function bus is closed")
        if name not in self._functions:
            raise FunctionNotFoundError(f"unknown function {name!r}")
        if allowed is not None and name not in allowed:
            raise FunctionPermissionError(f"actor {actor!r} cannot call {name!r}")

        function = self._functions[name]
        payload = dict(arguments)
        self._validate(name, function.input_schema or _OBJECT_SCHEMA, payload, "input")
        if function.mutates and not self._writes_open:
            raise DomainClosedError(f"mutation {name!r} rejected after Domain termination")

        if asyncio.current_task() is None:  # pragma: no cover - requires an event loop task
            raise RuntimeError("function calls require an asyncio task")
        call_id = next(self._ids)
        started_at = time.time()
        monotonic_start = time.monotonic()
        self._active_count += 1
        self._idle.clear()
        if function.mutates:
            self._active_mutation_count += 1
            self._mutations_idle.clear()

        context = CallContext(call_id=call_id, actor=actor, function=name)
        try:
            if function.serial_key is None:
                result = await self._invoke(function, context, payload)
            else:
                async with self._serial_locks[function.serial_key]:
                    if function.mutates and not self._writes_open:
                        raise DomainClosedError(
                            f"mutation {name!r} rejected after Domain termination"
                        )
                    result = await self._invoke(function, context, payload)
        except BaseException as error:
            self._records.append(
                CallRecord(
                    call_id=call_id,
                    started_at=started_at,
                    duration_s=time.monotonic() - monotonic_start,
                    actor=actor,
                    function=name,
                    arguments=_audit_value(payload),
                    outcome="cancelled" if isinstance(error, asyncio.CancelledError) else "error",
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        else:
            self._records.append(
                CallRecord(
                    call_id=call_id,
                    started_at=started_at,
                    duration_s=time.monotonic() - monotonic_start,
                    actor=actor,
                    function=name,
                    arguments=_audit_value(payload),
                    outcome="ok",
                    result=_audit_value(result),
                )
            )
            return result
        finally:
            self._active_count -= 1
            if self._active_count == 0:
                self._idle.set()
            if function.mutates:
                self._active_mutation_count -= 1
                if self._active_mutation_count == 0:
                    self._mutations_idle.set()

    async def _invoke(
        self, function: Function, context: CallContext, payload: JsonObject
    ) -> Any:
        operation = function.handler(context, payload)
        if not inspect.isawaitable(operation):
            raise ContractError(
                f"function {function.name!r} handler must return an awaitable"
            )
        result = await operation
        if function.output_schema is not None:
            self._validate(function.name, function.output_schema, result, "output")
        return result

    @staticmethod
    def _validate(name: str, schema: JsonObject, value: Any, direction: str) -> None:
        try:
            Draft202012Validator(schema).validate(value)
        except ValidationError as error:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            raise FunctionValidationError(
                f"{name} {direction} at {location}: {error.message}"
            ) from error


class FunctionClient:
    def __init__(
        self,
        bus: FunctionBus,
        actor: str,
        allowed: frozenset[str] | None,
    ) -> None:
        self._bus = bus
        self.actor = actor
        self._allowed = allowed

    @property
    def specs(self) -> tuple[FunctionSpec, ...]:
        return self._bus.specs(self._allowed)

    def has(self, name: str) -> bool:
        return name in {spec.name for spec in self.specs}

    async def call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload = dict(arguments or {})
        payload.update(kwargs)
        return await self._bus.call(self.actor, name, payload, self._allowed)


async def _wait(event: asyncio.Event, timeout_s: float | None) -> None:
    if timeout_s is None:
        await event.wait()
    else:
        await asyncio.wait_for(event.wait(), timeout=timeout_s)


def _audit_value(value: Any, depth: int = 0) -> Any:
    if depth >= 6:
        return {"type": type(value).__name__, "truncated": True}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _audit_value(item, depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple)):
        return [_audit_value(item, depth + 1) for item in value[:100]]
    if is_dataclass(value) and not isinstance(value, type):
        return _audit_value(asdict(value), depth + 1)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {
            "type": type(value).__name__,
            "shape": list(shape),
            "dtype": str(dtype),
        }
    return {"type": type(value).__name__}

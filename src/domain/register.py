from __future__ import annotations

import asyncio
import inspect
import re
import threading
import time
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Mapping

from domain.errors import DomainClosedError, RegisterError
from domain.io import json_value


_NAME = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
Handler = Callable[..., Any]
Getter = Callable[[], Any]
Setter = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    mutates: bool

    @property
    def provider_name(self) -> str:
        return self.name.replace(".", "__")

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.provider_name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "strict": False,
        }


@dataclass(frozen=True, slots=True)
class CallRecord:
    call_id: int
    time: float
    duration_s: float
    actor: str
    operation: str
    name: str
    arguments: Any
    outcome: str
    result: Any = None
    error: str | None = None


@dataclass(slots=True)
class _Function:
    owner: str
    handler: Handler
    spec: FunctionSpec
    serial_key: str | None


@dataclass(slots=True)
class _Reference:
    owner: str
    getter: Getter
    setter: Setter | None
    description: str


class DomainRegister:
    """Thread-safe, Domain-local directory of callable functions and references."""

    def __init__(self) -> None:
        self._functions: dict[str, _Function] = {}
        self._references: dict[str, _Reference] = {}
        self._serial_locks: dict[str, threading.Lock] = {}
        self._records: list[CallRecord] = []
        self._ids = count(1)
        self._condition = threading.Condition(threading.RLock())
        self._writes_open = True

    @property
    def names(self) -> frozenset[str]:
        with self._condition:
            return frozenset((*self._functions, *self._references))

    @property
    def records(self) -> tuple[CallRecord, ...]:
        with self._condition:
            return tuple(sorted(self._records, key=lambda item: item.call_id))

    def register_function(
        self,
        owner: str,
        name: str,
        handler: Handler,
        *,
        description: str,
        parameters: Mapping[str, Any] | None = None,
        mutates: bool = False,
        serial_key: str | None = None,
    ) -> None:
        self._validate_registration(owner, name)
        if not callable(handler):
            raise RegisterError(f"function {name!r} handler must be callable")
        if not description.strip():
            raise RegisterError(f"function {name!r} description must not be empty")
        if mutates and serial_key is None:
            serial_key = name
        entry = _Function(
            owner,
            handler,
            FunctionSpec(
                name,
                description,
                dict(parameters or {"type": "object", "additionalProperties": False}),
                mutates,
            ),
            serial_key,
        )
        with self._condition:
            self._require_free(name)
            self._functions[name] = entry
            if serial_key is not None:
                self._serial_locks.setdefault(serial_key, threading.Lock())
            self._condition.notify_all()

    def register_reference(
        self,
        owner: str,
        name: str,
        getter: Getter,
        *,
        setter: Setter | None = None,
        description: str = "",
    ) -> None:
        self._validate_registration(owner, name)
        if not callable(getter) or (setter is not None and not callable(setter)):
            raise RegisterError(f"reference {name!r} requires callable accessors")
        with self._condition:
            self._require_free(name)
            self._references[name] = _Reference(owner, getter, setter, description)
            self._condition.notify_all()

    def wait_for(self, name: str, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while name not in self._functions and name not in self._references:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def functions(self, prefix: str | None = None) -> tuple[FunctionSpec, ...]:
        with self._condition:
            values = (
                entry.spec
                for name, entry in self._functions.items()
                if prefix is None or name.startswith(prefix)
            )
            return tuple(sorted(values, key=lambda item: item.name))

    def openai_tools(self, prefix: str | None = None) -> list[dict[str, Any]]:
        return self.openai_toolset(prefix)[0]

    def openai_toolset(
        self, prefix: str | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        tools: list[dict[str, Any]] = []
        names: dict[str, str] = {}
        for spec in self.functions(prefix):
            if spec.provider_name in names:
                raise RegisterError(
                    f"function name collision after provider mapping: {spec.name!r}"
                )
            names[spec.provider_name] = spec.name
            tools.append(spec.as_openai_tool())
        return tools, names

    def call(
        self,
        actor: str,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        entry = self._function(name)
        self._check_write(entry)
        payload = dict(arguments or {})
        lock = self._lock_for(entry.serial_key)
        started = time.monotonic()
        call_id = next(self._ids)
        try:
            def invoke() -> Any:
                value = entry.handler(**payload)
                if not inspect.isawaitable(value):
                    return value
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(_await_value(value))
                value.close() if inspect.iscoroutine(value) else None
                raise RegisterError(
                    f"function {name!r} returned an awaitable; use await register.acall(...)"
                )

            if lock is None:
                value = invoke()
            else:
                with lock:
                    self._check_write(entry)
                    value = invoke()
        except BaseException as error:
            self._record(call_id, started, actor, "call", name, payload, None, error)
            raise
        self._record(call_id, started, actor, "call", name, payload, value, None)
        return value

    async def acall(
        self,
        actor: str,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        entry = self._function(name)
        self._check_write(entry)
        payload = dict(arguments or {})
        lock = self._lock_for(entry.serial_key)
        started = time.monotonic()
        call_id = next(self._ids)
        try:
            if lock is not None:
                await asyncio.to_thread(lock.acquire)
            try:
                self._check_write(entry)
                value = entry.handler(**payload)
                if inspect.isawaitable(value):
                    value = await value
            finally:
                if lock is not None:
                    lock.release()
        except BaseException as error:
            self._record(call_id, started, actor, "call", name, payload, None, error)
            raise
        self._record(call_id, started, actor, "call", name, payload, value, None)
        return value

    def read(self, actor: str, name: str) -> Any:
        entry = self._reference(name)
        started = time.monotonic()
        call_id = next(self._ids)
        try:
            value = entry.getter()
        except BaseException as error:
            self._record(call_id, started, actor, "read", name, {}, None, error)
            raise
        self._record(call_id, started, actor, "read", name, {}, value, None)
        return value

    def write(self, actor: str, name: str, value: Any) -> None:
        if not self._writes_open:
            raise DomainClosedError(f"reference write {name!r} rejected after termination")
        entry = self._reference(name)
        if entry.setter is None:
            raise RegisterError(f"reference {name!r} is read-only")
        started = time.monotonic()
        call_id = next(self._ids)
        try:
            entry.setter(value)
        except BaseException as error:
            self._record(call_id, started, actor, "write", name, value, None, error)
            raise
        self._record(call_id, started, actor, "write", name, value, None, None)

    def close_writes(self) -> None:
        with self._condition:
            self._writes_open = False

    def manifest(self) -> dict[str, Any]:
        with self._condition:
            return {
                "functions": {
                    name: {
                        "owner": entry.owner,
                        "description": entry.spec.description,
                        "parameters": dict(entry.spec.parameters),
                        "mutates": entry.spec.mutates,
                        "serial_key": entry.serial_key,
                    }
                    for name, entry in sorted(self._functions.items())
                },
                "references": {
                    name: {
                        "owner": entry.owner,
                        "description": entry.description,
                        "writable": entry.setter is not None,
                    }
                    for name, entry in sorted(self._references.items())
                },
            }

    def _function(self, name: str) -> _Function:
        with self._condition:
            try:
                return self._functions[name]
            except KeyError as error:
                raise RegisterError(f"unknown function {name!r}") from error

    def _reference(self, name: str) -> _Reference:
        with self._condition:
            try:
                return self._references[name]
            except KeyError as error:
                raise RegisterError(f"unknown reference {name!r}") from error

    def _check_write(self, entry: _Function) -> None:
        if entry.spec.mutates and not self._writes_open:
            raise DomainClosedError(
                f"mutation {entry.spec.name!r} rejected after Domain termination"
            )

    def _lock_for(self, key: str | None) -> threading.Lock | None:
        if key is None:
            return None
        with self._condition:
            return self._serial_locks[key]

    def _record(
        self,
        call_id: int,
        started: float,
        actor: str,
        operation: str,
        name: str,
        arguments: Any,
        result: Any,
        error: BaseException | None,
    ) -> None:
        record = CallRecord(
            call_id=call_id,
            time=time.time(),
            duration_s=time.monotonic() - started,
            actor=actor,
            operation=operation,
            name=name,
            arguments=json_value(arguments),
            outcome="error" if error else "ok",
            result=json_value(result),
            error=(f"{type(error).__name__}: {error}" if error else None),
        )
        with self._condition:
            self._records.append(record)

    def _validate_registration(self, owner: str, name: str) -> None:
        if not owner.strip():
            raise RegisterError("register owner must not be empty")
        if not _NAME.fullmatch(name):
            raise RegisterError(f"invalid register name {name!r}; use dotted lowercase names")

    def _require_free(self, name: str) -> None:
        if name in self._functions or name in self._references:
            raise RegisterError(f"register name {name!r} is already owned")


async def _await_value(value: Any) -> Any:
    return await value

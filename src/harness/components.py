from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.functions import Function, FunctionClient
from harness.output import ComponentOutput
from schemas import JsonObject, NavigationTask, Terminal


@dataclass(frozen=True, slots=True)
class ComponentContext:
    domain_id: str
    task: NavigationTask
    functions: FunctionClient
    output: ComponentOutput
    cancelled: asyncio.Event


class Component(ABC):
    name: str = ""
    required_functions: frozenset[str] = frozenset()
    optional_functions: frozenset[str] = frozenset()

    def functions(self) -> Sequence[Function]:
        return ()

    async def start(self, context: ComponentContext) -> None:
        pass

    async def close(self, reason: str) -> None:
        pass


class Agent(Component, ABC):
    @abstractmethod
    async def run(self, context: ComponentContext) -> None:
        raise NotImplementedError


class Environment(Component, ABC):
    @abstractmethod
    async def wait_terminal(self) -> Terminal:
        raise NotImplementedError

    @abstractmethod
    async def result(self) -> JsonObject:
        raise NotImplementedError


class Metric(Component, ABC):
    @abstractmethod
    async def evaluate(
        self, terminal: Terminal, environment: Mapping[str, Any]
    ) -> Mapping[str, float]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DomainComponents:
    environment: Environment
    agent: Agent
    services: tuple[Component, ...] = field(default_factory=tuple)
    metrics: tuple[Metric, ...] = field(default_factory=tuple)

    @property
    def all(self) -> tuple[Component, ...]:
        return (self.environment, *self.services, *self.metrics, self.agent)

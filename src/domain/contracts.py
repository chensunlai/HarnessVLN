from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from domain.errors import HarnessError


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class NavigationEpisode:
    episode_id: str
    instruction: str
    scene_id: str | None = None
    public: Mapping[str, Any] = field(default_factory=dict)
    setup: Mapping[str, Any] = field(default_factory=dict)
    truth: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise HarnessError("episode_id must not be empty")
        if not self.instruction.strip():
            raise HarnessError("instruction must not be empty")
        object.__setattr__(self, "public", dict(self.public))
        object.__setattr__(self, "setup", dict(self.setup))
        object.__setattr__(self, "truth", dict(self.truth))

    def as_dict(self, *, include_truth: bool = True) -> JsonObject:
        value: JsonObject = {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "scene_id": self.scene_id,
            "public": dict(self.public),
            "setup": dict(self.setup),
        }
        if include_truth:
            value["truth"] = dict(self.truth)
        return value


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    name: str
    factory: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise HarnessError("module name must not be empty")
        module, separator, attribute = self.factory.partition(":")
        if not separator or not module.strip() or not attribute.strip():
            raise HarnessError(
                f"module factory {self.factory!r} must use 'package:attribute' syntax"
            )
        object.__setattr__(self, "params", dict(self.params))

    def as_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "factory": self.factory,
            "params": dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class WorkspaceSpec:
    python: str = "python"
    environment: Mapping[str, str] = field(default_factory=dict)
    command_timeout_s: float = 30.0
    max_output_chars: int = 200_000

    def __post_init__(self) -> None:
        if not self.python.strip():
            raise HarnessError("workspace python command must not be empty")
        if self.command_timeout_s <= 0 or not math.isfinite(self.command_timeout_s):
            raise HarnessError("workspace command_timeout_s must be positive")
        if self.max_output_chars < 1:
            raise HarnessError("workspace max_output_chars must be positive")
        object.__setattr__(
            self,
            "environment",
            {str(key): str(value) for key, value in self.environment.items()},
        )

    def as_dict(self) -> JsonObject:
        return {
            "python": self.python,
            "environment": dict(self.environment),
            "command_timeout_s": self.command_timeout_s,
            "max_output_chars": self.max_output_chars,
        }


@dataclass(frozen=True, slots=True)
class DomainSpec:
    environment: ModuleSpec
    metric: ModuleSpec
    modules: tuple[ModuleSpec, ...] = ()
    workspace: WorkspaceSpec = field(default_factory=WorkspaceSpec)
    timeout_s: float = 300.0
    shutdown_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        names = [self.environment.name, self.metric.name, *(item.name for item in self.modules)]
        if self.environment.name != "env":
            raise HarnessError("the environment module must be named 'env'")
        if self.metric.name != "metric":
            raise HarnessError("the metric module must be named 'metric'")
        if len(set(names)) != len(names):
            raise HarnessError("Domain module names must be unique")
        if self.timeout_s <= 0 or self.shutdown_timeout_s <= 0:
            raise HarnessError("Domain timeouts must be positive")
        object.__setattr__(self, "modules", tuple(self.modules))

    @property
    def all_modules(self) -> tuple[ModuleSpec, ...]:
        return (self.environment, self.metric, *self.modules)

    def as_dict(self) -> JsonObject:
        return {
            "environment": self.environment.as_dict(),
            "metric": self.metric.as_dict(),
            "modules": [item.as_dict() for item in self.modules],
            "workspace": self.workspace.as_dict(),
            "timeout_s": self.timeout_s,
            "shutdown_timeout_s": self.shutdown_timeout_s,
        }


@dataclass(frozen=True, slots=True)
class Terminal:
    status: str
    reason: str
    actor: str

    def as_dict(self) -> JsonObject:
        return {"status": self.status, "reason": self.reason, "actor": self.actor}


@dataclass(frozen=True, slots=True)
class DomainResult:
    domain_id: str
    episode_id: str
    terminal: Terminal
    environment: Mapping[str, Any]
    metrics: Mapping[str, float]
    modules: Mapping[str, Mapping[str, Any]]
    errors: tuple[str, ...] = ()
    workspace: str | None = None

    def as_dict(self) -> JsonObject:
        return {
            "schema_version": 1,
            "domain_id": self.domain_id,
            "episode_id": self.episode_id,
            "terminal": self.terminal.as_dict(),
            "environment": dict(self.environment),
            "metrics": dict(self.metrics),
            "modules": {name: dict(value) for name, value in self.modules.items()},
            "errors": list(self.errors),
            "workspace": self.workspace,
        }


@dataclass(frozen=True, slots=True)
class DomainJob:
    index: int
    bench_id: str
    episode: NavigationEpisode
    spec: DomainSpec
    output_dir: Path
    domain_id: str

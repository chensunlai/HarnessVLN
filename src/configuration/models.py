from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from domain.contracts import DomainSpec, ModuleSpec, WorkspaceSpec
from domain.errors import HarnessError


@dataclass(frozen=True, slots=True)
class FactorySpec:
    factory: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"factory": self.factory, "params": dict(self.params)}


@dataclass(frozen=True, slots=True)
class DomainTemplate:
    modules: tuple[ModuleSpec, ...]
    workspace: WorkspaceSpec
    timeout_s: float
    shutdown_timeout_s: float

    def bind(self, environment: ModuleSpec, metric: ModuleSpec) -> DomainSpec:
        return DomainSpec(
            environment,
            metric,
            self.modules,
            self.workspace,
            self.timeout_s,
            self.shutdown_timeout_s,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "modules": [module.as_dict() for module in self.modules],
            "workspace": self.workspace.as_dict(),
            "timeout_s": self.timeout_s,
            "shutdown_timeout_s": self.shutdown_timeout_s,
        }


@dataclass(frozen=True, slots=True)
class BenchConfig:
    bench_id: str
    benchmark: FactorySpec
    environment: ModuleSpec
    metric: ModuleSpec
    max_episodes: int | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        if not self.bench_id.strip():
            raise HarnessError("bench id must not be empty")
        if self.max_episodes is not None and self.max_episodes < 1:
            raise HarnessError("bench max_episodes must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.bench_id,
            "benchmark": self.benchmark.as_dict(),
            "environment": self.environment.as_dict(),
            "metric": self.metric.as_dict(),
            "max_episodes": self.max_episodes,
            "source": str(self.source) if self.source else None,
        }


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    name: str
    device: int | None = None
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise HarnessError("worker name must not be empty")
        if self.device is not None and self.device < 0:
            raise HarnessError("worker device must not be negative")
        object.__setattr__(
            self,
            "environment",
            {str(key): str(value) for key, value in self.environment.items()},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    domain: DomainTemplate
    benches: tuple[BenchConfig, ...]
    workers: tuple[WorkerConfig, ...]
    output_root: Path
    run_id: str | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        if not self.benches:
            raise HarnessError("runner requires at least one bench")
        if not self.workers:
            raise HarnessError("runner requires at least one worker")
        if len({item.bench_id for item in self.benches}) != len(self.benches):
            raise HarnessError("runner bench ids must be unique")
        if len({item.name for item in self.workers}) != len(self.workers):
            raise HarnessError("runner worker names must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.as_dict(),
            "benches": [bench.as_dict() for bench in self.benches],
            "workers": [worker.as_dict() for worker in self.workers],
            "output_root": str(self.output_root),
            "run_id": self.run_id,
            "source": str(self.source) if self.source else None,
        }

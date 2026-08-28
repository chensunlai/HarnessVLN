from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from harness.errors import ContractError
from schemas import JsonObject


@dataclass(frozen=True, slots=True)
class FactorySpec:
    target: str
    parameters: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        module, separator, attribute = self.target.partition(":")
        if not separator or not module.strip() or not attribute.strip():
            raise ContractError(
                f"factory target {self.target!r} must use 'module:attribute' syntax"
            )
        object.__setattr__(self, "parameters", dict(self.parameters))

    def as_dict(self) -> JsonObject:
        return {"factory": self.target, "parameters": dict(self.parameters)}


@dataclass(frozen=True, slots=True)
class AgentConfig:
    agent: FactorySpec
    components: tuple[FactorySpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.agent, FactorySpec):
            raise TypeError("agent must be a FactorySpec")
        if any(not isinstance(component, FactorySpec) for component in self.components):
            raise TypeError("components must contain FactorySpec values")
        object.__setattr__(self, "components", tuple(self.components))

    def as_dict(self) -> JsonObject:
        return {
            "agent": self.agent.as_dict(),
            "components": [component.as_dict() for component in self.components],
        }


@dataclass(frozen=True, slots=True)
class BenchConfig:
    config_id: str
    benchmark: FactorySpec
    environment: FactorySpec
    metrics: tuple[FactorySpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.config_id.strip():
            raise ContractError("Bench configuration ID must not be empty")
        if not isinstance(self.benchmark, FactorySpec):
            raise TypeError("benchmark must be a FactorySpec")
        if not isinstance(self.environment, FactorySpec):
            raise TypeError("environment must be a FactorySpec")
        if any(not isinstance(metric, FactorySpec) for metric in self.metrics):
            raise TypeError("metrics must contain FactorySpec values")
        object.__setattr__(self, "metrics", tuple(self.metrics))

    def as_dict(self) -> JsonObject:
        return {
            "id": self.config_id,
            "benchmark": self.benchmark.as_dict(),
            "environment": self.environment.as_dict(),
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    name: str
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ContractError("worker name must not be empty")
        object.__setattr__(
            self,
            "environment",
            {str(name): str(value) for name, value in self.environment.items()},
        )

    def as_dict(self) -> JsonObject:
        return {"name": self.name, "environment": dict(self.environment)}


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    agent: AgentConfig
    benches: tuple[BenchConfig, ...]
    output_dir: Path
    workers: tuple[WorkerConfig, ...]
    timeout_s: float = 300.0
    shutdown_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.benches:
            raise ContractError("RunnerConfig requires at least one Bench")
        if not self.workers:
            raise ContractError("RunnerConfig requires at least one worker")
        if len({bench.config_id for bench in self.benches}) != len(self.benches):
            raise ContractError("Bench configuration IDs must be unique")
        if len({worker.name for worker in self.workers}) != len(self.workers):
            raise ContractError("worker names must be unique")
        if (
            self.timeout_s <= 0
            or self.shutdown_timeout_s <= 0
            or not math.isfinite(self.timeout_s)
            or not math.isfinite(self.shutdown_timeout_s)
        ):
            raise ContractError("Runner timeouts must be positive finite numbers")
        object.__setattr__(self, "benches", tuple(self.benches))
        object.__setattr__(self, "workers", tuple(self.workers))
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())

    def as_dict(self) -> JsonObject:
        return {
            "agent": self.agent.as_dict(),
            "benches": [bench.as_dict() for bench in self.benches],
            "output_dir": str(self.output_dir),
            "workers": [worker.as_dict() for worker in self.workers],
            "timeout_s": self.timeout_s,
            "shutdown_timeout_s": self.shutdown_timeout_s,
        }


def load_runner_config(path: str | Path) -> RunnerConfig:
    source = Path(path).expanduser().resolve()
    data = _mapping(_read_yaml(source), source)
    _keys(
        data,
        {"agent", "benches", "output_dir", "parallelism", "workers", "timeout_s", "shutdown_timeout_s"},
        source,
    )
    agent_ref = _required(data, "agent", source)
    if not isinstance(agent_ref, str):
        raise ContractError(f"{source}: agent must reference an agent YAML file")
    agent = load_agent_config(_reference(source, agent_ref))

    bench_refs = _sequence(_required(data, "benches", source), source, "benches")
    if not bench_refs:
        raise ContractError(f"{source}: benches must not be empty")
    benches = tuple(
        load_bench_config(_reference(source, _string(ref, source, "bench reference")))
        for ref in bench_refs
    )
    bench_ids = [bench.config_id for bench in benches]
    if len(set(bench_ids)) != len(bench_ids):
        raise ContractError(f"{source}: Bench configuration IDs must be unique")

    parallelism = int(data.get("parallelism", 1))
    if parallelism < 1:
        raise ContractError(f"{source}: parallelism must be positive")
    raw_workers = data.get("workers")
    if raw_workers is None:
        workers = tuple(WorkerConfig(f"worker-{index}") for index in range(parallelism))
    else:
        worker_values = _sequence(raw_workers, source, "workers")
        workers = tuple(
            _worker(value, source, index) for index, value in enumerate(worker_values)
        )
        if len(workers) != parallelism:
            raise ContractError(
                f"{source}: workers count must equal parallelism ({parallelism})"
            )
    worker_names = [worker.name for worker in workers]
    if len(set(worker_names)) != len(worker_names):
        raise ContractError(f"{source}: worker names must be unique")

    output_value = _string(data.get("output_dir", "../../runs"), source, "output_dir")
    output_dir = _reference(source, output_value)
    timeout_s = float(data.get("timeout_s", 300.0))
    shutdown_timeout_s = float(data.get("shutdown_timeout_s", 10.0))
    if (
        timeout_s <= 0
        or shutdown_timeout_s <= 0
        or not math.isfinite(timeout_s)
        or not math.isfinite(shutdown_timeout_s)
    ):
        raise ContractError(f"{source}: timeouts must be positive finite numbers")
    return RunnerConfig(
        agent=agent,
        benches=benches,
        output_dir=output_dir,
        workers=workers,
        timeout_s=timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
    )


def load_agent_config(path: str | Path) -> AgentConfig:
    source = Path(path).expanduser().resolve()
    data = _mapping(_read_yaml(source), source)
    _keys(data, {"agent", "components"}, source)
    agent = _factory(_required(data, "agent", source), source)
    components = tuple(
        _factory(value, source)
        for value in _sequence(data.get("components", ()), source, "components")
    )
    return AgentConfig(agent=agent, components=components)


def load_bench_config(path: str | Path) -> BenchConfig:
    source = Path(path).expanduser().resolve()
    data = _mapping(_read_yaml(source), source)
    _keys(data, {"id", "benchmark", "environment", "metrics"}, source)
    config_id = _string(data.get("id", source.stem), source, "id")
    benchmark = _factory(_required(data, "benchmark", source), source)
    environment = _factory(_required(data, "environment", source), source)
    metrics = tuple(
        _factory(value, source)
        for value in _sequence(data.get("metrics", ()), source, "metrics")
    )
    return BenchConfig(config_id, benchmark, environment, metrics)


def _factory(value: Any, source: Path) -> FactorySpec:
    if isinstance(value, str):
        referenced = _reference(source, value)
        return _factory(_read_yaml(referenced), referenced)
    data = _mapping(value, source)
    _keys(data, {"factory", "parameters"}, source)
    target = _string(_required(data, "factory", source), source, "factory")
    parameters = _mapping(data.get("parameters", {}), source)
    return FactorySpec(target, dict(parameters))


def _worker(value: Any, source: Path, index: int) -> WorkerConfig:
    data = _mapping(value, source)
    _keys(data, {"name", "environment"}, source)
    name = _string(data.get("name", f"worker-{index}"), source, "worker name")
    raw_environment = _mapping(data.get("environment", {}), source)
    environment = {str(key): str(item) for key, item in raw_environment.items()}
    return WorkerConfig(name, environment)


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise ContractError(f"configuration file does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ContractError(f"invalid YAML in {path}: {error}") from error


def _reference(source: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def _mapping(value: Any, source: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{source}: expected a mapping")
    return value


def _sequence(value: Any, source: Path, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{source}: {field_name} must be a list")
    return value


def _string(value: Any, source: Path, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{source}: {field_name} must be a non-empty string")
    return value


def _required(data: Mapping[str, Any], name: str, source: Path) -> Any:
    if name not in data:
        raise ContractError(f"{source}: missing required field {name!r}")
    return data[name]


def _keys(data: Mapping[str, Any], allowed: set[str], source: Path) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ContractError(f"{source}: unknown fields: {', '.join(unknown)}")

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from configuration.models import (
    BenchConfig,
    DomainTemplate,
    FactorySpec,
    RunnerConfig,
    WorkerConfig,
)
from domain.contracts import ModuleSpec, WorkspaceSpec
from domain.errors import HarnessError


def load_runner(path: str | Path) -> RunnerConfig:
    source = Path(path).expanduser().resolve()
    root = _section(source, "runner")
    _keys(root, {"domain", "benches", "output_root", "run_id", "workers"}, source)
    domain_ref = _reference(source, _string(_required(root, "domain", source), source, "domain"))
    domain = load_domain(domain_ref)
    bench_values = _sequence(_required(root, "benches", source), source, "benches")
    benches = tuple(
        load_bench(_reference(source, _string(value, source, "bench reference")))
        for value in bench_values
    )
    workers = _workers(root.get("workers", {"count": 1}), source)
    output_value = _string(root.get("output_root", "../../runs"), source, "output_root")
    output_root = _reference(source, os.path.expandvars(output_value))
    run_id = root.get("run_id")
    if run_id is not None:
        run_id = _string(run_id, source, "run_id")
    return RunnerConfig(domain, benches, workers, output_root, run_id, source)


def load_domain(path: str | Path) -> DomainTemplate:
    source = Path(path).expanduser().resolve()
    root = _section(source, "domain")
    _keys(root, {"modules", "workspace", "timeout_s", "shutdown_timeout_s"}, source)
    modules = tuple(
        load_module(_reference(source, _string(value, source, "module reference")))
        for value in _sequence(root.get("modules", ()), source, "modules")
    )
    workspace_value = _mapping(root.get("workspace", {}), source, "workspace")
    _keys(
        workspace_value,
        {"python", "environment", "command_timeout_s", "max_output_chars"},
        source,
    )
    environment = _mapping(
        workspace_value.get("environment", {}), source, "workspace.environment"
    )
    workspace = WorkspaceSpec(
        python=str(workspace_value.get("python", "python")),
        environment={str(key): str(value) for key, value in environment.items()},
        command_timeout_s=float(workspace_value.get("command_timeout_s", 30.0)),
        max_output_chars=int(workspace_value.get("max_output_chars", 200_000)),
    )
    return DomainTemplate(
        modules,
        workspace,
        float(root.get("timeout_s", 300.0)),
        float(root.get("shutdown_timeout_s", 10.0)),
    )


def load_bench(path: str | Path) -> BenchConfig:
    source = Path(path).expanduser().resolve()
    root = _section(source, "bench")
    _keys(
        root,
        {"id", "factory", "params", "environment", "metric", "max_episodes"},
        source,
    )
    bench_id = _string(root.get("id", source.stem), source, "bench.id")
    factory = _string(_required(root, "factory", source), source, "bench.factory")
    params = _mapping(root.get("params", {}), source, "bench.params")
    environment = load_environment(
        _reference(
            source,
            _string(_required(root, "environment", source), source, "bench.environment"),
        )
    )
    metric = load_metric(
        _reference(source, _string(_required(root, "metric", source), source, "bench.metric"))
    )
    max_episodes = root.get("max_episodes")
    return BenchConfig(
        bench_id,
        FactorySpec(factory, dict(params)),
        environment,
        metric,
        int(max_episodes) if max_episodes is not None else None,
        source,
    )


def load_module(path: str | Path) -> ModuleSpec:
    source = Path(path).expanduser().resolve()
    root = _section(source, "module")
    return _module_spec(root, source, default_name=source.stem)


def load_environment(path: str | Path) -> ModuleSpec:
    source = Path(path).expanduser().resolve()
    root = _section(source, "environment")
    return _module_spec(root, source, default_name="env", fixed_name="env")


def load_metric(path: str | Path) -> ModuleSpec:
    source = Path(path).expanduser().resolve()
    root = _section(source, "metric")
    return _module_spec(root, source, default_name="metric", fixed_name="metric")


def _module_spec(
    root: Mapping[str, Any],
    source: Path,
    *,
    default_name: str,
    fixed_name: str | None = None,
) -> ModuleSpec:
    _keys(root, {"name", "factory", "params"}, source)
    name = _string(root.get("name", default_name), source, "module.name")
    if fixed_name is not None and name != fixed_name:
        raise HarnessError(f"{source}: module must be named {fixed_name!r}")
    factory = _string(_required(root, "factory", source), source, "module.factory")
    params = _mapping(root.get("params", {}), source, "module.params")
    return ModuleSpec(name, factory, dict(params))


def _workers(value: Any, source: Path) -> tuple[WorkerConfig, ...]:
    data = _mapping(value, source, "workers")
    _keys(data, {"count", "devices", "per_device", "environment"}, source)
    environment = {
        str(key): str(item)
        for key, item in _mapping(data.get("environment", {}), source, "workers.environment").items()
    }
    devices_value = data.get("devices")
    if devices_value is not None:
        devices = [int(item) for item in _sequence(devices_value, source, "workers.devices")]
        if not devices or len(set(devices)) != len(devices) or any(item < 0 for item in devices):
            raise HarnessError(f"{source}: worker devices must be unique non-negative ids")
        per_device = int(data.get("per_device", 1))
        if per_device < 1:
            raise HarnessError(f"{source}: workers.per_device must be positive")
        return tuple(
            WorkerConfig(f"gpu-{device}-{slot}", device, environment)
            for device in devices
            for slot in range(per_device)
        )
    count = int(data.get("count", 1))
    if count < 1:
        raise HarnessError(f"{source}: workers.count must be positive")
    if "per_device" in data:
        raise HarnessError(f"{source}: workers.per_device requires workers.devices")
    return tuple(WorkerConfig(f"cpu-{index}", None, environment) for index in range(count))


def _section(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HarnessError(f"failed to load config {path}: {error}") from error
    document = _mapping(value, path, "document")
    _keys(document, {name}, path)
    return _mapping(_required(document, name, path), path, name)


def _reference(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (owner.parent / path).resolve()


def _mapping(value: Any, source: Path, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessError(f"{source}: {field} must be an object")
    return value


def _sequence(value: Any, source: Path, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HarnessError(f"{source}: {field} must be a list")
    return value


def _string(value: Any, source: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{source}: {field} must be a non-empty string")
    return value


def _required(value: Mapping[str, Any], key: str, source: Path) -> Any:
    if key not in value:
        raise HarnessError(f"{source}: missing required field {key!r}")
    return value[key]


def _keys(value: Mapping[str, Any], allowed: set[str], source: Path) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise HarnessError(f"{source}: unknown fields: {', '.join(unknown)}")

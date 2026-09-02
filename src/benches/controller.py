from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

from benches.base import Benchmark
from configuration.models import BenchConfig, DomainTemplate
from domain.contracts import DomainJob, DomainResult
from domain.errors import HarnessError
from domain.io import write_json


@dataclass(frozen=True, slots=True)
class DomainRecord:
    index: int
    bench_id: str
    episode_id: str
    worker: str
    worker_pid: int
    device: int | None
    duration_s: float
    result: DomainResult | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "bench_id": self.bench_id,
            "episode_id": self.episode_id,
            "worker": self.worker,
            "worker_pid": self.worker_pid,
            "device": self.device,
            "duration_s": self.duration_s,
            "result": self.result.as_dict() if self.result else None,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BenchSummary:
    bench_id: str
    name: str
    split: str
    records: tuple[DomainRecord, ...]
    metrics: Mapping[str, float]
    error: str | None = None

    @property
    def failed(self) -> bool:
        return bool(
            self.error
            or any(
                record.error
                or record.result is None
                or record.result.terminal.status not in {"completed", "environment_terminal"}
                or record.result.errors
                for record in self.records
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "bench_id": self.bench_id,
            "name": self.name,
            "split": self.split,
            "failed": self.failed,
            "metrics": dict(self.metrics),
            "error": self.error,
            "counts": {
                "episodes": len(self.records),
                "errors": sum(record.error is not None for record in self.records),
                "task_failures": sum(
                    record.result is not None
                    and record.result.terminal.status not in {"completed", "environment_terminal"}
                    for record in self.records
                ),
            },
            "episodes": [record.as_dict() for record in self.records],
        }


class BenchmarkController:
    """Own one benchmark's episodes, Env/Metric selection, and aggregation."""

    def __init__(
        self,
        config: BenchConfig,
        domain: DomainTemplate,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.domain = domain
        self.output_dir = output_dir
        self.benchmark = _benchmark(config)
        output_dir.mkdir(parents=True)
        write_json(output_dir / "config.json", config.as_dict())

    def jobs(self) -> list[DomainJob]:
        episodes = self.benchmark.episodes()
        if self.config.max_episodes is not None:
            episodes = islice(episodes, self.config.max_episodes)
        spec = self.domain.bind(self.config.environment, self.config.metric)
        return [
            DomainJob(
                index,
                self.config.bench_id,
                episode,
                spec,
                self.output_dir / "episodes",
                f"{index:06d}-{_slug(episode.episode_id)}",
            )
            for index, episode in enumerate(episodes)
        ]

    def finish(self, records: tuple[DomainRecord, ...]) -> BenchSummary:
        results = tuple(record.result for record in records if record.result is not None)
        error: str | None = None
        metrics: dict[str, float] = {}
        try:
            metrics = dict(self.benchmark.aggregate(results))
        except BaseException as exception:
            error = f"{type(exception).__name__}: {exception}"
        summary = BenchSummary(
            self.config.bench_id,
            self.benchmark.name,
            self.benchmark.split,
            records,
            metrics,
            error,
        )
        write_json(self.output_dir / "result.json", summary)
        return summary


def _benchmark(config: BenchConfig) -> Benchmark:
    module_name, separator, attribute = config.benchmark.factory.partition(":")
    if not separator:
        raise HarnessError(f"invalid benchmark factory: {config.benchmark.factory}")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        value = factory(**dict(config.benchmark.params))
    except BaseException as error:
        raise HarnessError(
            f"benchmark {config.bench_id!r} factory failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(value, Benchmark):
        raise HarnessError(
            f"benchmark factory returned {type(value).__name__}, expected Benchmark"
        )
    return value


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not result:
        raise HarnessError(f"identifier cannot form a path: {value!r}")
    return result

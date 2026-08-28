from __future__ import annotations

import asyncio
import multiprocessing
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benches import Benchmark, BenchmarkCase
from harness.components import Agent, Component, DomainComponents, Environment, Metric
from harness.config import BenchConfig, FactorySpec, RunnerConfig, WorkerConfig
from harness.domain import DomainResult, DomainRuntime
from harness.factory import instantiate
from harness.output import write_json
from schemas import JsonObject


@dataclass(frozen=True, slots=True)
class DomainJob:
    index: int
    bench_id: str
    case: BenchmarkCase
    agent: FactorySpec
    environment: FactorySpec
    services: tuple[FactorySpec, ...]
    metrics: tuple[FactorySpec, ...]
    timeout_s: float
    shutdown_timeout_s: float
    output_root: str
    domain_id: str


@dataclass(frozen=True, slots=True)
class DomainRecord:
    index: int
    bench_id: str
    case_id: str
    worker: str
    worker_pid: int
    result: DomainResult | None = None
    error: str | None = None

    def as_dict(self) -> JsonObject:
        return {
            "index": self.index,
            "bench_id": self.bench_id,
            "case_id": self.case_id,
            "worker": self.worker,
            "worker_pid": self.worker_pid,
            "result": self.result.as_dict() if self.result else None,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BenchSummary:
    bench_id: str
    name: str
    split: str
    records: tuple[DomainRecord, ...]
    aggregate_metrics: dict[str, float]
    error: str | None = None

    def as_dict(self) -> JsonObject:
        return {
            "bench_id": self.bench_id,
            "name": self.name,
            "split": self.split,
            "aggregate_metrics": dict(self.aggregate_metrics),
            "error": self.error,
            "records": [record.as_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    output_dir: str
    benches: tuple[BenchSummary, ...]

    @property
    def failed(self) -> bool:
        return any(
            bench.error
            or any(
                record.error
                or record.result is None
                or record.result.terminal.status not in {"completed", "environment_terminal"}
                or record.result.cleanup_errors
                for record in bench.records
            )
            for bench in self.benches
        )

    def as_dict(self) -> JsonObject:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "failed": self.failed,
            "benches": [bench.as_dict() for bench in self.benches],
        }


CompletedCallback = Callable[[DomainRecord, int, int], None]


class Runner:
    """Control-plane scheduler for complete Domain jobs."""

    def run(
        self,
        config: RunnerConfig,
        *,
        run_id: str | None = None,
        on_completed: CompletedCallback | None = None,
    ) -> RunSummary:
        identifier = _run_id(run_id)
        run_root = config.output_dir / identifier
        configured_benches: list[tuple[int, BenchConfig, Benchmark]] = []
        jobs: list[DomainJob] = []

        for bench_index, bench_config in enumerate(config.benches):
            benchmark = instantiate(
                bench_config.benchmark, Benchmark, "benchmark"  # type: ignore[type-abstract]
            )
            configured_benches.append((bench_index, bench_config, benchmark))
            bench_dir = (
                run_root
                / "benches"
                / f"{bench_index:03d}-{_slug(bench_config.config_id)}"
            )
            for case_index, case in enumerate(benchmark.cases()):
                domain_id = f"{case_index:06d}-{_slug(case.case_id)}"
                jobs.append(
                    DomainJob(
                        index=len(jobs),
                        bench_id=bench_config.config_id,
                        case=case,
                        agent=config.agent.agent,
                        environment=bench_config.environment,
                        services=config.agent.components,
                        metrics=bench_config.metrics,
                        timeout_s=config.timeout_s,
                        shutdown_timeout_s=config.shutdown_timeout_s,
                        output_root=str(bench_dir / "episodes"),
                        domain_id=domain_id,
                    )
                )

        run_root.mkdir(parents=True, exist_ok=True)
        write_json(run_root / "config.json", config.as_dict())
        records = self._execute(jobs, config.workers, on_completed)
        by_bench: dict[str, list[DomainRecord]] = {}
        for record in records:
            by_bench.setdefault(record.bench_id, []).append(record)

        summaries: list[BenchSummary] = []
        for bench_index, bench_config, benchmark in configured_benches:
            bench_records = tuple(
                sorted(by_bench.get(bench_config.config_id, ()), key=lambda item: item.index)
            )
            successful_results = tuple(
                record.result for record in bench_records if record.result is not None
            )
            aggregate: dict[str, float] = {}
            error: str | None = None
            try:
                aggregate = dict(benchmark.aggregate(successful_results))
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
            summary = BenchSummary(
                bench_id=bench_config.config_id,
                name=benchmark.name,
                split=benchmark.split,
                records=bench_records,
                aggregate_metrics=aggregate,
                error=error,
            )
            summaries.append(summary)
            bench_dir = (
                run_root
                / "benches"
                / f"{bench_index:03d}-{_slug(bench_config.config_id)}"
            )
            write_json(bench_dir / "result.json", summary.as_dict())

        result = RunSummary(identifier, str(run_root), tuple(summaries))
        write_json(run_root / "result.json", result.as_dict())
        return result

    def _execute(
        self,
        jobs: Sequence[DomainJob],
        workers: Sequence[WorkerConfig],
        on_completed: CompletedCallback | None,
    ) -> list[DomainRecord]:
        if not jobs:
            return []
        context = multiprocessing.get_context("spawn")
        executors = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(worker.environment,),
            )
            for worker in workers
        ]
        records: list[DomainRecord] = []
        iterator = iter(jobs)
        active: dict[Future[DomainRecord], tuple[int, DomainJob]] = {}
        completed = 0
        try:
            for slot, executor in enumerate(executors):
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                active[executor.submit(_execute_job, job, workers[slot].name)] = (slot, job)

            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    slot, job = active.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        record = DomainRecord(
                            index=job.index,
                            bench_id=job.bench_id,
                            case_id=job.case.case_id,
                            worker=workers[slot].name,
                            worker_pid=-1,
                            error=f"worker failed: {type(error).__name__}: {error}",
                        )
                    records.append(record)
                    completed += 1
                    if on_completed is not None:
                        on_completed(record, completed, len(jobs))
                    try:
                        next_job = next(iterator)
                    except StopIteration:
                        continue
                    active[
                        executors[slot].submit(_execute_job, next_job, workers[slot].name)
                    ] = (slot, next_job)
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
        return sorted(records, key=lambda record: record.index)


def _initialize_worker(environment: Mapping[str, str]) -> None:
    for name, value in environment.items():
        os.environ[name] = value


def _execute_job(job: DomainJob, worker: str) -> DomainRecord:
    try:
        environment = instantiate(
            job.environment, Environment, "environment"  # type: ignore[type-abstract]
        )
        agent = instantiate(job.agent, Agent, "agent")  # type: ignore[type-abstract]
        services = tuple(
            instantiate(spec, Component, f"service {index}")
            for index, spec in enumerate(job.services)
        )
        metrics = tuple(
            instantiate(
                spec, Metric, f"metric {index}"  # type: ignore[type-abstract]
            )
            for index, spec in enumerate(job.metrics)
        )
        result = asyncio.run(
            DomainRuntime(job.timeout_s, job.shutdown_timeout_s).run(
                job.case.task,
                DomainComponents(environment, agent, services, metrics),
                output_root=job.output_root,
                domain_id=job.domain_id,
            )
        )
        return DomainRecord(
            index=job.index,
            bench_id=job.bench_id,
            case_id=job.case.case_id,
            worker=worker,
            worker_pid=os.getpid(),
            result=result,
        )
    except Exception as error:
        return DomainRecord(
            index=job.index,
            bench_id=job.bench_id,
            case_id=job.case.case_id,
            worker=worker,
            worker_pid=os.getpid(),
            error=f"{type(error).__name__}: {error}",
        )


def _run_id(value: str | None) -> str:
    if value is not None:
        return _slug(value)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        raise ValueError(f"identifier {value!r} cannot form a file name")
    return slug

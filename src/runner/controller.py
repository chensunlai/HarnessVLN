from __future__ import annotations

import multiprocessing
import re
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any

from benches.controller import BenchmarkController, BenchSummary, DomainRecord
from configuration.models import RunnerConfig, WorkerConfig
from domain.contracts import DomainJob
from domain.errors import HarnessError
from domain.io import write_json
from runner.contracts import RunSummary
from runner.worker import execute_job, initialize_worker


class Runner:
    """Schedule complete Domain jobs. It never observes or acts in an environment."""

    def run(self, config: RunnerConfig, *, progress: bool = True) -> RunSummary:
        run_id = _run_id(config.run_id)
        run_root = config.output_root / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        write_json(run_root / "config" / "resolved.json", config.as_dict())

        controllers, jobs = self._jobs(config, run_root)
        tracker = _Progress(len(jobs), enabled=progress)
        try:
            records = self._execute(jobs, config.workers, tracker)
        finally:
            tracker.close()

        by_bench: dict[str, list[DomainRecord]] = {}
        for record in records:
            by_bench.setdefault(record.bench_id, []).append(record)

        summaries: list[BenchSummary] = []
        for controller in controllers:
            bench_records = tuple(
                sorted(
                    by_bench.get(controller.config.bench_id, ()),
                    key=lambda item: item.index,
                )
            )
            summaries.append(controller.finish(bench_records))

        result = RunSummary(run_id, str(run_root), tuple(summaries))
        write_json(run_root / "result.json", result)
        return result

    def _jobs(
        self, config: RunnerConfig, run_root: Path
    ) -> tuple[list[BenchmarkController], list[DomainJob]]:
        controllers: list[BenchmarkController] = []
        bench_jobs: list[list[DomainJob]] = []
        for bench_index, bench_config in enumerate(config.benches):
            bench_dir = (
                run_root
                / "benches"
                / f"{bench_index:03d}-{_slug(bench_config.bench_id)}"
            )
            controller = BenchmarkController(bench_config, config.domain, bench_dir)
            controllers.append(controller)
            current_jobs = controller.jobs()
            bench_jobs.append(current_jobs)
        jobs: list[DomainJob] = []
        iterators = [iter(items) for items in bench_jobs]
        while iterators:
            remaining = []
            for iterator in iterators:
                try:
                    job = next(iterator)
                except StopIteration:
                    continue
                jobs.append(replace(job, index=len(jobs)))
                remaining.append(iterator)
            iterators = remaining
        return controllers, jobs

    @staticmethod
    def _execute(
        jobs: list[DomainJob],
        workers: tuple[WorkerConfig, ...],
        tracker: _Progress,
    ) -> list[DomainRecord]:
        if not jobs:
            return []
        context = multiprocessing.get_context("spawn")
        executors = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=initialize_worker,
                initargs=(worker,),
            )
            for worker in workers
        ]
        iterator = iter(jobs)
        active: dict[Future[DomainRecord], tuple[int, DomainJob]] = {}
        records: list[DomainRecord] = []
        aborted = False
        try:
            for slot, executor in enumerate(executors):
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                active[executor.submit(execute_job, job)] = (slot, job)
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                released_slots: list[int] = []
                execution_failed = False
                for future in done:
                    slot, job = active.pop(future)
                    try:
                        record = future.result()
                    except BaseException as error:
                        record = DomainRecord(
                            job.index,
                            job.bench_id,
                            job.episode.episode_id,
                            workers[slot].name,
                            -1,
                            workers[slot].device,
                            0.0,
                            error=f"worker failed: {type(error).__name__}: {error}",
                        )
                    records.append(record)
                    tracker.update(record)
                    released_slots.append(slot)
                    execution_failed = execution_failed or bool(
                        _execution_errors(record)
                    )

                if execution_failed and not aborted:
                    aborted = True
                    tracker.abort(len(jobs) - len(records) - len(active))
                if aborted:
                    continue

                for slot in released_slots:
                    try:
                        next_job = next(iterator)
                    except StopIteration:
                        continue
                    active[executors[slot].submit(execute_job, next_job)] = (
                        slot,
                        next_job,
                    )
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
        return sorted(records, key=lambda item: item.index)


class _Progress:
    def __init__(self, total: int, *, enabled: bool) -> None:
        self._bar: Any = None
        if enabled:
            try:
                from tqdm import tqdm

                self._bar = tqdm(
                    total=total, desc="episodes", dynamic_ncols=True, position=0
                )
            except ImportError:
                self._bar = None
        self._completed = 0
        self._total = total
        self._metric_totals: dict[str, dict[str, float]] = {}
        self._metric_counts: dict[str, dict[str, int]] = {}

    def update(self, record: DomainRecord) -> None:
        self._completed += 1
        errors = _execution_errors(record)
        if record.result is not None and not errors:
            totals = self._metric_totals.setdefault(record.bench_id, {})
            counts = self._metric_counts.setdefault(record.bench_id, {})
            for name, value in record.result.metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1
        if self._bar is not None:
            self._bar.set_postfix(self._metric_postfix(record.bench_id), refresh=False)
            self._bar.update(1)
        for error in errors:
            self._write(
                f"ERROR [{self._completed}/{self._total}] "
                f"bench={record.bench_id} episode={record.episode_id} "
                f"worker={record.worker}: {error}"
            )

    def abort(self, not_started: int) -> None:
        self._write(
            "ERROR batch aborted after an execution failure; "
            f"{max(0, not_started)} episodes were not started"
        )

    def _write(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message, file=self._bar.fp)
        else:
            print(message, file=sys.stderr, flush=True)

    def _metric_postfix(self, bench_id: str) -> dict[str, str]:
        totals = self._metric_totals.get(bench_id, {})
        counts = self._metric_counts.get(bench_id, {})
        priority = {name: index for index, name in enumerate(_METRIC_PRIORITY)}
        names = sorted(
            totals, key=lambda name: (priority.get(name, len(priority)), name)
        )
        values = {
            f"avg_{name}": f"{totals[name] / counts[name]:.3f}"
            for name in names[:4]
            if counts.get(name, 0)
        }
        if len(self._metric_totals) > 1:
            return {"bench": bench_id, **values}
        return values

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _execution_errors(record: DomainRecord) -> tuple[str, ...]:
    errors: list[str] = []
    if record.error:
        errors.append(record.error)
    if record.result is None:
        if not errors:
            errors.append("worker returned no Domain result")
    else:
        errors.extend(record.result.errors)
    return tuple(errors)


_METRIC_PRIORITY = ("sr", "spl", "ne", "os", "success", "path_efficiency")


def _run_id(value: str | None) -> str:
    if value:
        return _slug(value)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not result:
        raise HarnessError(f"identifier cannot form a path: {value!r}")
    return result

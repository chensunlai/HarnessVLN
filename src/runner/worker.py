from __future__ import annotations

import os
import time

from benches.controller import DomainRecord
from configuration.models import WorkerConfig
from domain.contracts import DomainJob
from domain.runtime import DomainRuntime


_WORKER: WorkerConfig | None = None


def initialize_worker(worker: WorkerConfig) -> None:
    global _WORKER
    _WORKER = worker
    for name, value in worker.environment.items():
        os.environ[name] = value
    os.environ["HARNESS_WORKER"] = worker.name
    if worker.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device)
        os.environ["HARNESS_DEVICE"] = str(worker.device)


def execute_job(job: DomainJob) -> DomainRecord:
    worker = _WORKER or WorkerConfig("local")
    started = time.monotonic()
    try:
        result = DomainRuntime().run(
            job.episode,
            job.spec,
            job.output_dir,
            domain_id=job.domain_id,
        )
        return DomainRecord(
            job.index,
            job.bench_id,
            job.episode.episode_id,
            worker.name,
            os.getpid(),
            worker.device,
            time.monotonic() - started,
            result=result,
        )
    except BaseException as error:
        return DomainRecord(
            job.index,
            job.bench_id,
            job.episode.episode_id,
            worker.name,
            os.getpid(),
            worker.device,
            time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
        )

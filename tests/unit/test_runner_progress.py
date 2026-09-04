from benches.controller import DomainRecord
from domain import DomainResult, Terminal
from runner.controller import _Progress


def record(
    index: int,
    bench_id: str,
    metrics: dict[str, float],
    *,
    errors: tuple[str, ...] = (),
) -> DomainRecord:
    result = DomainResult(
        f"domain-{index}",
        f"episode-{index}",
        Terminal("completed", "done", "agent"),
        {},
        metrics,
        {},
        errors,
    )
    return DomainRecord(
        index,
        bench_id,
        result.episode_id,
        "worker",
        1,
        None,
        1.0,
        result=result,
    )


def test_progress_reports_per_bench_running_metric_averages() -> None:
    progress = _Progress(4, enabled=False)
    progress.update(record(0, "r2r", {"sr": 1.0, "spl": 0.5, "ne": 2.0}))
    progress.update(record(1, "r2r", {"sr": 0.0, "spl": 0.0, "ne": 6.0}))
    progress.update(record(2, "other", {"sr": 1.0}))
    progress.update(
        record(3, "r2r", {"sr": 0.0, "spl": 0.0, "ne": 100.0}, errors=("bad",))
    )

    assert progress._metric_postfix("r2r") == {
        "bench": "r2r",
        "avg_sr": "0.500",
        "avg_spl": "0.250",
        "avg_ne": "4.000",
    }
    assert progress._metric_postfix("other") == {
        "bench": "other",
        "avg_sr": "1.000",
    }

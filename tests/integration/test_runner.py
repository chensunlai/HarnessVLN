import json
from pathlib import Path

from harness.config import BenchConfig, FactorySpec, RunnerConfig, load_runner_config
from harness.runner import Runner


ROOT = Path(__file__).resolve().parents[2]


def test_runner_executes_complete_domains_in_fixed_worker_processes(tmp_path):
    loaded = load_runner_config(ROOT / "config" / "runners" / "dummy.yaml")
    second_bench = BenchConfig(
        config_id="dummy_second",
        benchmark=FactorySpec(
            "benches.dummy:DummyBenchmark", {"targets": [5]}
        ),
        environment=loaded.benches[0].environment,
        metrics=loaded.benches[0].metrics,
    )
    config = RunnerConfig(
        agent=loaded.agent,
        benches=(*loaded.benches, second_bench),
        output_dir=tmp_path,
        workers=loaded.workers,
        timeout_s=loaded.timeout_s,
        shutdown_timeout_s=loaded.shutdown_timeout_s,
    )
    progress = []

    summary = Runner().run(
        config,
        run_id="parallel-test",
        on_completed=lambda record, completed, total: progress.append(
            (record.case_id, completed, total)
        ),
    )

    assert not summary.failed
    assert len(summary.benches) == 2
    bench = summary.benches[0]
    assert len(bench.records) == 6
    assert bench.aggregate_metrics == {
        "success": 1.0,
        "spl": 1.0,
        "distance": 0.0,
    }
    assert {record.worker for record in bench.records} == {"local-0", "local-1"}
    assert len({record.worker_pid for record in bench.records}) == 2
    assert summary.benches[1].aggregate_metrics == {
        "success": 1.0,
        "spl": 1.0,
        "distance": 0.0,
    }
    assert len(progress) == 7
    assert progress[-1][1:] == (7, 7)

    run_root = tmp_path / "parallel-test"
    run_result = json.loads((run_root / "result.json").read_text())
    assert run_result["failed"] is False
    assert (run_root / "config.json").is_file()
    bench_root = run_root / "benches" / "000-dummy"
    assert (bench_root / "result.json").is_file()
    assert len(list((bench_root / "episodes").iterdir())) == 6
    assert (run_root / "benches" / "001-dummy_second" / "result.json").is_file()

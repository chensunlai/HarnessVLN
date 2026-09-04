from __future__ import annotations

from dataclasses import replace

from configuration import load_runner
from domain import ModuleSpec
from runner import Runner


def test_runner_uses_multiple_process_workers(tmp_path) -> None:
    config = load_runner("config/runners/dummy.yaml")
    config = replace(config, output_root=tmp_path, run_id="runner-test")
    result = Runner().run(config, progress=False)

    assert not result.failed
    assert len(result.benches) == 1
    assert len(result.benches[0].records) == 4
    assert len({item.worker_pid for item in result.benches[0].records}) == 2
    assert result.benches[0].metrics["success"] == 1.0
    assert result.metrics["success"] == 1.0
    assert (tmp_path / "runner-test/result.json").is_file()


def test_runner_reports_execution_errors_and_stops_scheduling(tmp_path, capsys) -> None:
    config = load_runner("config/runners/dummy.yaml")
    domain = replace(
        config.domain,
        modules=(
            ModuleSpec(
                "failing",
                "tests.fixtures.service_modules:FailingModule",
            ),
        ),
    )
    config = replace(
        config,
        domain=domain,
        output_root=tmp_path,
        run_id="runner-failure-test",
    )

    result = Runner().run(config, progress=True)

    records = result.benches[0].records
    assert result.failed
    assert len(records) == len(config.workers)
    assert all(record.result and record.result.errors for record in records)
    assert result.benches[0].as_dict()["counts"]["errors"] == len(records)
    error_output = capsys.readouterr().err
    assert "ERROR" in error_output
    assert "backend unavailable" in error_output
    assert "2 episodes were not started" in error_output

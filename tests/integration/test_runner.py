from __future__ import annotations

from dataclasses import replace

from configuration import load_runner
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

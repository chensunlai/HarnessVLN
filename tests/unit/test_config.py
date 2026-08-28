from pathlib import Path

import pytest

from harness.config import load_runner_config
from harness.errors import ContractError


ROOT = Path(__file__).resolve().parents[2]


def test_runner_config_resolves_agent_bench_environment_and_metric_references():
    config = load_runner_config(ROOT / "config" / "runners" / "dummy.yaml")

    assert config.agent.agent.target == "agents.dummy:DummyAgent"
    assert [component.target for component in config.agent.components] == [
        "memory.dummy:DummyMemory",
        "vln.dummy:DummyVLN",
    ]
    assert config.benches[0].benchmark.target == "benches.dummy:DummyBenchmark"
    assert config.benches[0].environment.target == "envs.dummy:DummyEnvironment"
    assert config.benches[0].metrics[0].target == "metrics.dummy:DummyMetric"
    assert [worker.name for worker in config.workers] == ["local-0", "local-1"]


def test_runner_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "runner.yaml"
    path.write_text(
        "agent: agent.yaml\nbenches: [bench.yaml]\nparallelism: 1\nunexpected: true\n"
    )
    with pytest.raises(ContractError, match="unknown fields: unexpected"):
        load_runner_config(path)

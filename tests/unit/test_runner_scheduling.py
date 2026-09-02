from dataclasses import replace

from configuration import load_runner
from runner import Runner


def test_bench_episode_jobs_are_interleaved(tmp_path) -> None:
    config = load_runner("config/runners/dummy.yaml")
    first = config.benches[0]
    second = replace(first, bench_id="dummy-second")
    config = replace(config, benches=(first, second))

    _, jobs = Runner()._jobs(config, tmp_path)

    assert [job.bench_id for job in jobs[:4]] == [
        "dummy",
        "dummy-second",
        "dummy",
        "dummy-second",
    ]

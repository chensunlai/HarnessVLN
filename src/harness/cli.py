from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from harness.config import load_runner_config
from harness.runner import DomainRecord, Runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.cli")
    parser.add_argument(
        "--runner-config",
        required=True,
        help="Runner YAML; it references Bench and Agent configurations.",
    )
    parser.add_argument("--run-id", help="Optional deterministic output directory name.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_runner_config(arguments.runner_config)

    def progress(record: DomainRecord, completed: int, total: int) -> None:
        status = record.result.terminal.status if record.result else "error"
        print(
            f"\r[{completed}/{total}] {record.bench_id}/{record.case_id}: {status}",
            end="",
            flush=True,
        )

    summary = Runner().run(config, run_id=arguments.run_id, on_completed=progress)
    if any(bench.records for bench in summary.benches):
        print()
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "output_dir": summary.output_dir,
                "failed": summary.failed,
                "benchmarks": {
                    bench.bench_id: bench.aggregate_metrics for bench in summary.benches
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

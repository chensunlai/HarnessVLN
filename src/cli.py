from __future__ import annotations

import argparse
from collections.abc import Sequence

from configuration import load_runner
from runner import Runner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cli")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run all benches referenced by a runner config")
    run.add_argument("--runner", required=True, metavar="PATH")
    run.add_argument("--no-progress", action="store_true")
    arguments = parser.parse_args(argv)
    config = load_runner(arguments.runner)
    summary = Runner().run(config, progress=not arguments.no_progress)
    for bench in summary.benches:
        print(
            f"{bench.bench_id}: {len(bench.records)} episodes, "
            f"failed={bench.failed}, metrics={dict(bench.metrics)}"
        )
    print(summary.output_dir)
    return int(summary.failed)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from benches import Benchmark, BenchmarkCase
from harness import NavigationTask


class DummyBenchmark(Benchmark):
    name = "dummy_navigation"
    split = "test"

    def __init__(self, targets=(2, -3, 0)) -> None:
        self.targets = tuple(int(target) for target in targets)

    def cases(self):
        for index, target in enumerate(self.targets):
            yield BenchmarkCase(
                case_id=f"dummy-{index}",
                task=NavigationTask(
                    task_id=f"dummy-{index}",
                    instruction=f"Navigate to position {target}",
                    metadata={"target": target},
                ),
            )

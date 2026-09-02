from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benches.controller import BenchSummary


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    output_dir: str
    benches: tuple[BenchSummary, ...]

    @property
    def failed(self) -> bool:
        return any(bench.failed for bench in self.benches)

    @property
    def metrics(self) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for bench in self.benches:
            for record in bench.records:
                if record.result is None:
                    continue
                for name, value in record.result.metrics.items():
                    values.setdefault(name, []).append(float(value))
        return {
            name: sum(items) / len(items)
            for name, items in sorted(values.items())
            if items
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "failed": self.failed,
            "metrics": self.metrics,
            "benches": [bench.as_dict() for bench in self.benches],
        }

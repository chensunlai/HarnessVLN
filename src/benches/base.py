from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from harness.domain import DomainResult
from schemas import NavigationTask


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    task: NavigationTask


class Benchmark(ABC):
    name: str
    split: str

    @abstractmethod
    def cases(self) -> Iterable[BenchmarkCase]:
        raise NotImplementedError

    def aggregate(self, results: Sequence[DomainResult]) -> Mapping[str, float]:
        values: dict[str, list[float]] = {}
        for result in results:
            for name, value in result.metrics.items():
                values.setdefault(name, []).append(value)
        return {
            name: sum(items) / len(items)
            for name, items in values.items()
            if items
        }

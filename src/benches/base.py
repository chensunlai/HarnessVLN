from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

from domain.contracts import DomainResult, NavigationEpisode


class Benchmark(ABC):
    name: str
    split: str

    @abstractmethod
    def episodes(self) -> Iterable[NavigationEpisode]:
        raise NotImplementedError

    def aggregate(self, results: Sequence[DomainResult]) -> Mapping[str, float]:
        values: dict[str, list[float]] = {}
        for result in results:
            for name, value in result.metrics.items():
                values.setdefault(name, []).append(float(value))
        return {
            name: sum(items) / len(items)
            for name, items in sorted(values.items())
            if items
        }

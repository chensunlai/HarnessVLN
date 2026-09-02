from __future__ import annotations

from collections.abc import Iterable, Sequence

from benches.base import Benchmark
from domain.contracts import NavigationEpisode


class DummyBenchmark(Benchmark):
    name = "dummy_navigation"

    def __init__(
        self,
        trajectories: Sequence[Sequence[str]] | None = None,
        *,
        split: str = "smoke",
    ) -> None:
        self.split = split
        self.trajectories = tuple(
            tuple(actions)
            for actions in (
                trajectories
                or (
                    ("forward", "forward", "left", "forward"),
                    ("right", "forward"),
                    ("left", "left", "forward"),
                    ("forward",),
                )
            )
        )

    def episodes(self) -> Iterable[NavigationEpisode]:
        for index, actions in enumerate(self.trajectories):
            episode_id = f"dummy:{self.split}:{index}"
            yield NavigationEpisode(
                episode_id,
                {
                    "type": "instruction",
                    "instruction": "Follow the reference route to the destination.",
                },
                scene_id="dummy-grid",
                public={"split": self.split},
                setup={"start_pose": [0, 0, 0]},
                truth={"expert_actions": list(actions)},
            )

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from benches.base import Benchmark
from benches.io import load_json, require_fields
from domain.contracts import NavigationEpisode
from domain.errors import HarnessError


class RoboTHORObjectNavBenchmark(Benchmark):
    name = "robothor_objectnav_2021"

    def __init__(self, root: str | Path, *, split: str = "val") -> None:
        self.root = Path(root)
        self.split = split

    def episodes(self) -> Iterable[NavigationEpisode]:
        directory = self.root / self.split / "episodes"
        paths = sorted(directory.glob("*.json.gz"))
        if not paths:
            raise HarnessError(f"RoboTHOR episode files not found under {directory}")
        seen: set[str] = set()
        for path in paths:
            values = load_json(path)
            if not isinstance(values, list):
                raise HarnessError(f"RoboTHOR file must contain a list: {path}")
            for value in values:
                raw = require_fields(
                    value,
                    {
                        "id",
                        "scene",
                        "object_type",
                        "initial_position",
                        "initial_orientation",
                        "initial_horizon",
                    },
                    "RoboTHOR episode",
                )
                episode_id = f"robothor:{self.split}:{raw['id']}"
                if episode_id in seen:
                    raise HarnessError(f"duplicate RoboTHOR episode: {episode_id}")
                seen.add(episode_id)
                category = str(raw["object_type"])
                yield NavigationEpisode(
                    episode_id,
                    f"Find the {category}.",
                    str(raw["scene"]),
                    {"split": self.split, "category": category},
                    {
                        "source_episode_id": raw["id"],
                        "scene": raw["scene"],
                        "initial_position": raw["initial_position"],
                        "initial_orientation": raw["initial_orientation"],
                        "initial_horizon": raw["initial_horizon"],
                        "object_type": category,
                    },
                    {
                        key: raw[key]
                        for key in ("shortest_path", "shortest_path_length")
                        if key in raw
                    },
                )

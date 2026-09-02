from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benches.base import Benchmark
from benches.io import load_json, require_fields
from domain.contracts import NavigationEpisode
from domain.errors import HarnessError


class HabitatObjectNavBenchmark(Benchmark):
    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str,
        split: str = "val",
    ) -> None:
        if dataset not in {"mp3d", "hm3d"}:
            raise ValueError("dataset must be 'mp3d' or 'hm3d'")
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.name = f"habitat_objectnav_{dataset}"

    def episodes(self) -> Iterable[NavigationEpisode]:
        directory = self.root / self.split / "content"
        paths = sorted(directory.glob("*.json.gz"))
        if not paths:
            raise HarnessError(f"ObjectNav content shards not found under {directory}")
        seen: set[str] = set()
        for path in paths:
            document = load_json(path)
            if not isinstance(document, dict):
                raise HarnessError(f"invalid ObjectNav shard: {path}")
            episodes = document.get("episodes")
            goals = document.get("goals_by_category")
            if not isinstance(episodes, list) or not isinstance(goals, dict):
                raise HarnessError(f"invalid ObjectNav shard structure: {path}")
            scene_key = path.name.removesuffix(".json.gz")
            for index, raw_value in enumerate(episodes):
                raw = require_fields(
                    raw_value,
                    {
                        "episode_id",
                        "scene_id",
                        "start_position",
                        "start_rotation",
                        "object_category",
                        "info",
                    },
                    "ObjectNav episode",
                )
                category = str(raw["object_category"])
                goal_key = f"{Path(str(raw['scene_id'])).name}_{category}"
                if not isinstance(goals.get(goal_key), list) or not goals[goal_key]:
                    raise HarnessError(f"ObjectNav goal table has no key {goal_key}")
                episode_id = f"objectnav:{self.dataset}:{self.split}:{scene_key}:{index}"
                if episode_id in seen:
                    raise HarnessError(f"duplicate ObjectNav episode: {episode_id}")
                seen.add(episode_id)
                setup: dict[str, Any] = {
                    "source_episode_id": raw["episode_id"],
                    "scene_id": raw["scene_id"],
                    "start_position": raw["start_position"],
                    "start_rotation": raw["start_rotation"],
                    "object_category": category,
                    "native_episode_index": index,
                }
                for name in ("scene_dataset_config", "additional_obj_config_paths"):
                    if name in raw:
                        setup[name] = raw[name]
                yield NavigationEpisode(
                    episode_id,
                    {"type": "target_text", "instruction": category},
                    str(raw["scene_id"]),
                    {"split": self.split, "dataset": self.dataset, "category": category},
                    setup,
                    {"goals": goals[goal_key], "info": raw["info"]},
                )

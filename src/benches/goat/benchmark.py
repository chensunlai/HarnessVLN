from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benches.base import Benchmark
from benches.io import load_json, require_fields
from domain.contracts import NavigationEpisode
from domain.errors import HarnessError


class GOATBenchmark(Benchmark):
    name = "goat_bench"

    def __init__(self, root: str | Path, *, split: str = "val_unseen") -> None:
        self.root = Path(root)
        self.split = split

    def episodes(self) -> Iterable[NavigationEpisode]:
        directory = self.root / self.split / "content"
        paths = sorted(directory.glob("*.json.gz"))
        if not paths:
            raise HarnessError(f"GOAT content shards not found under {directory}")
        seen: set[str] = set()
        for path in paths:
            document = load_json(path)
            if not isinstance(document, dict):
                raise HarnessError(f"invalid GOAT shard: {path}")
            episodes, table = document.get("episodes"), document.get("goals")
            if not isinstance(episodes, list) or not isinstance(table, dict):
                raise HarnessError(f"invalid GOAT shard structure: {path}")
            scene_key = path.name.removesuffix(".json.gz")
            for raw_value in episodes:
                raw = require_fields(
                    raw_value,
                    {"episode_id", "scene_id", "start_position", "start_rotation", "tasks"},
                    "GOAT episode",
                )
                episode_id = f"goat:{self.split}:{scene_key}:{raw['episode_id']}"
                if episode_id in seen:
                    raise HarnessError(f"duplicate GOAT episode: {episode_id}")
                seen.add(episode_id)
                goals, truth = _goals(episode_id, str(raw["scene_id"]), raw["tasks"], table)
                yield NavigationEpisode(
                    episode_id,
                    {"type": "goals", "goals": [dict(goal) for goal in goals]},
                    str(raw["scene_id"]),
                    {"split": self.split, "goal_count": len(goals), "goals": goals},
                    {
                        "source_episode_id": raw["episode_id"],
                        "scene_id": raw["scene_id"],
                        "scene_dataset_config": raw.get("scene_dataset_config"),
                        "start_position": raw["start_position"],
                        "start_rotation": raw["start_rotation"],
                        "native_tasks": raw["tasks"],
                        "goal_stream": goals,
                    },
                    {"goals": truth},
                )


def _goals(
    episode_id: str,
    scene_id: str,
    tasks: Any,
    table: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(tasks, list) or not tasks:
        raise HarnessError(f"GOAT episode {episode_id} has no tasks")
    scene = Path(scene_id).name
    public: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for index, native in enumerate(tasks):
        if not isinstance(native, list) or len(native) < 3:
            raise HarnessError(f"invalid GOAT task in {episode_id}: {native!r}")
        category, modality, object_id = native[:3]
        key = f"{scene}_{category}"
        instances = table.get(key)
        if not isinstance(instances, list) or not instances:
            raise HarnessError(f"GOAT goal table has no key {key}")
        selected = next(
            (item for item in instances if item.get("object_id") == object_id), None
        )
        instruction: dict[str, Any]
        if modality == "object":
            instruction = {"type": "target_text", "instruction": str(category)}
        elif modality == "description":
            text = str((selected or {}).get("lang_desc", "")).strip()
            if not text:
                raise HarnessError(f"GOAT task {index} in {episode_id} has no instruction")
            instruction = {"type": "target_text", "instruction": text}
        elif modality == "image":
            instruction = {"type": "target_img", "image": None}
        else:
            raise HarnessError(f"unknown GOAT modality {modality!r}")
        goal_id = f"{episode_id}:goal:{index}"
        public.append(instruction)
        truth.append(
            {
                "goal_id": goal_id,
                "goal_key": key,
                "modality": modality,
                "category": category,
                "object_id": object_id,
                "native": native,
            }
        )
    return public, truth

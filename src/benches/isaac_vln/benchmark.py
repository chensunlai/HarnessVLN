from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benches.base import Benchmark
from benches.io import load_json, require_fields
from domain.contracts import NavigationEpisode
from domain.errors import HarnessError


class IsaacVLNBenchmark(Benchmark):
    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str,
        split: str = "val_unseen",
        instruction_type: str = "formal",
        filter_same_trajectory: bool = False,
        filter_stairs: bool = False,
    ) -> None:
        if dataset not in {"vln_pe", "vlnverse"}:
            raise ValueError("dataset must be 'vln_pe' or 'vlnverse'")
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.instruction_type = instruction_type
        self.filter_same_trajectory = filter_same_trajectory
        self.filter_stairs = filter_stairs
        self.name = dataset

    @property
    def dataset_path(self) -> Path:
        candidates = (
            self.root / self.split / f"{self.split}.json.gz",
            self.root / f"{self.split}.json.gz",
            self.root / self.split / f"{self.split}.json",
        )
        for path in candidates:
            if path.exists():
                return path
        raise HarnessError(f"{self.dataset} split file not found under {self.root}")

    def episodes(self) -> Iterable[NavigationEpisode]:
        document = load_json(self.dataset_path)
        values = document.get("episodes") if isinstance(document, dict) else document
        if not isinstance(values, list):
            raise HarnessError(f"invalid {self.dataset} dataset: {self.dataset_path}")
        seen_trajectories: set[str] = set()
        seen: set[str] = set()
        indexed = sorted(enumerate(values), key=lambda item: (_scene(item[1]), item[0]))
        for _, raw_value in indexed:
            raw = require_fields(
                raw_value,
                {"trajectory_id", "episode_id", "instruction", "start_position", "start_rotation"},
                f"{self.dataset} episode",
            )
            trajectory = str(raw["trajectory_id"])
            if self.filter_same_trajectory and trajectory in seen_trajectories:
                continue
            seen_trajectories.add(trajectory)
            if self.filter_stairs and _vertical(raw):
                continue
            episode_id = f"{self.dataset}:{self.split}:{trajectory}_{raw['episode_id']}"
            if episode_id in seen:
                raise HarnessError(f"duplicate {self.dataset} episode: {episode_id}")
            seen.add(episode_id)
            instruction = _instruction(raw, self.instruction_type)
            scene_id = _scene(raw)
            yield NavigationEpisode(
                episode_id,
                instruction,
                scene_id,
                {"split": self.split, "dataset": self.dataset},
                {
                    "dataset_root": str(self.root),
                    "dataset_type": self.dataset,
                    "path_key": f"{trajectory}_{raw['episode_id']}",
                    "native_episode": raw,
                },
                {key: raw[key] for key in ("reference_path", "goals", "info") if key in raw},
            )


def _instruction(raw: Mapping[str, Any], kind: str) -> dict[str, Any]:
    value = raw["instruction"]
    if isinstance(value, str):
        if value.strip():
            return {"type": "instruction", "instruction": value.strip()}
    if isinstance(value, dict):
        for key in (kind, "instruction_text", "text", "formal"):
            if isinstance(value.get(key), str) and value[key].strip():
                return {"type": "instruction", "instruction": value[key].strip()}
    raise HarnessError(f"episode {raw.get('episode_id')} has no {kind} instruction")


def _scene(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    for key in ("scene_id", "scene", "scan"):
        if key in raw:
            return str(raw[key])
    return str(raw.get("trajectory_id", ""))


def _vertical(raw: Mapping[str, Any]) -> bool:
    path = raw.get("reference_path")
    if not isinstance(path, list) or len(path) < 2:
        return False
    try:
        heights = [float(point[1]) for point in path]
    except (IndexError, TypeError, ValueError):
        return False
    return max(heights) - min(heights) > 1.0

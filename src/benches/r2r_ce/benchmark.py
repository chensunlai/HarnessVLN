from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benches.base import Benchmark
from benches.io import load_json, require_fields
from domain.contracts import NavigationEpisode
from domain.errors import HarnessError


class R2RCEBenchmark(Benchmark):
    name = "r2r_ce"

    def __init__(self, root: str | Path, *, split: str = "val_unseen") -> None:
        self.root = Path(root)
        self.split = split

    @property
    def dataset_path(self) -> Path:
        directory = self.root / self.split
        for suffix in ("json.gz", "json"):
            path = directory / f"{self.split}.{suffix}"
            if path.exists():
                return path
        raise HarnessError(f"R2R-CE split file not found under {directory}")

    def episodes(self) -> Iterable[NavigationEpisode]:
        document = load_json(self.dataset_path)
        if not isinstance(document, dict) or not isinstance(document.get("episodes"), list):
            raise HarnessError(f"invalid R2R-CE dataset: {self.dataset_path}")
        for raw_value in document["episodes"]:
            raw = require_fields(
                raw_value,
                {"episode_id", "scene_id", "start_position", "start_rotation", "instruction"},
                "R2R-CE episode",
            )
            instruction_value = raw["instruction"]
            instruction = (
                instruction_value.get("instruction_text")
                if isinstance(instruction_value, dict)
                else instruction_value
            )
            if not isinstance(instruction, str) or not instruction.strip():
                raise HarnessError(f"R2R-CE episode {raw['episode_id']} has no instruction")
            episode_id = f"r2r_ce:{self.split}:{raw['episode_id']}"
            setup: dict[str, Any] = {
                "source_episode_id": raw["episode_id"],
                "scene_id": raw["scene_id"],
                "start_position": raw["start_position"],
                "start_rotation": raw["start_rotation"],
            }
            truth = {
                key: raw[key]
                for key in ("goals", "reference_path", "info")
                if key in raw
            }
            yield NavigationEpisode(
                episode_id,
                instruction.strip(),
                str(raw["scene_id"]),
                {"split": self.split},
                setup,
                truth,
            )

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.datasets.agent_vln.build import (
    PATTERNS,
    Candidate,
    InstructionVariant,
    _load_json,
    _manifest_episode,
    _native_episode,
    _stable_key,
    _write_gzip_json,
    _write_json,
    collect_candidates,
    validate_navmeshes,
)


def _read_manifest(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping) or not isinstance(value.get("episodes"), list):
        raise ValueError(f"{path} is not an AgentVLN candidate manifest")
    return value


def _read_split(root: Path, split: str) -> Mapping[str, Any]:
    path = root / split / f"{split}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping) or not isinstance(value.get("episodes"), list):
        raise ValueError(f"{path} is not a dataset split")
    return value


def _primary(candidate: Candidate, seed: int) -> InstructionVariant:
    eligible = [variant for variant in candidate.variants if variant.word_count <= 40]
    if not eligible:
        raise ValueError(
            f"trajectory {candidate.trajectory_id} has no bounded instruction"
        )
    return min(
        eligible,
        key=lambda value: (
            value.word_count,
            _stable_key(seed, "reserve-instruction", value.source_episode_id),
        ),
    )


def select_reserves(
    candidates: Sequence[Candidate],
    base_routes: Sequence[Mapping[str, Any]],
    *,
    per_pattern: int,
    max_per_scene: int,
    seed: int,
) -> dict[str, list[Candidate]]:
    scene_split: dict[str, str] = {}
    for route in base_routes:
        scene = str(route["scene_id"])
        split = str(route["split"])
        if scene in scene_split and scene_split[scene] != split:
            raise ValueError(f"base manifest leaks scene {scene} across splits")
        scene_split[scene] = split
    used = {str(route["source"]["trajectory_id"]) for route in base_routes}
    scene_counts = Counter(str(route["scene_id"]) for route in base_routes)
    selected: dict[str, list[Candidate]] = {
        split: [] for split in sorted(set(scene_split.values()))
    }
    remaining = [
        candidate
        for candidate in candidates
        if candidate.trajectory_id not in used
    ]

    for split in selected:
        for pattern in PATTERNS:
            for index in range(per_pattern):
                eligible = [
                    candidate
                    for candidate in remaining
                    if scene_split.get(candidate.scene_id) == split
                    and candidate.pattern == pattern
                    and scene_counts[candidate.scene_id] < max_per_scene
                    and any(
                        variant.word_count <= 40 for variant in candidate.variants
                    )
                ]
                if not eligible:
                    raise ValueError(
                        f"cannot fill reserve split={split}, pattern={pattern}; "
                        "adjust the distance or per-scene bound"
                    )
                candidate = min(
                    eligible,
                    key=lambda value: (
                        scene_counts[value.scene_id],
                        _stable_key(
                            seed,
                            f"reserve:{split}:{pattern}:{index}",
                            value.trajectory_id,
                        ),
                    ),
                )
                selected[split].append(candidate)
                scene_counts[candidate.scene_id] += 1
                remaining.remove(candidate)
    return selected


def extend(args: argparse.Namespace) -> Mapping[str, Any]:
    base_root = args.base.resolve()
    output_root = args.output.resolve()
    base_manifest = _read_manifest(base_root / "manifest.json")
    base_routes = list(base_manifest["episodes"])
    source = _load_json(args.source.resolve())
    candidates, geometry_rejected = collect_candidates(
        source, max_distance_m=args.max_distance
    )
    candidates, navmesh_rejected = validate_navmeshes(
        candidates,
        args.scenes_root.resolve(),
        max_distance_m=args.max_distance,
    )
    reserves = select_reserves(
        candidates,
        base_routes,
        per_pattern=args.per_pattern,
        max_per_scene=args.max_per_scene,
        seed=args.seed,
    )

    combined_routes = list(base_routes)
    split_documents = {
        split: _read_split(base_root, split) for split in sorted(reserves)
    }
    for split, values in reserves.items():
        for index, candidate in enumerate(values):
            primary = _primary(candidate, args.seed)
            route_id = (
                f"agent_vln:{split}:reserve_{candidate.pattern}_{index:02d}"
            )
            native = _native_episode(index, split, candidate, primary)
            record = _manifest_episode(index, split, candidate, primary)
            native["episode_id"] = route_id
            record["episode_id"] = route_id
            split_documents[split]["episodes"].append(native)
            combined_routes.append(record)

    for split, document in split_documents.items():
        _write_gzip_json(output_root / split / f"{split}.json.gz", document)
    manifest = {
        "schema_version": 1,
        "name": "agent_vln_r2r_local_candidate_pool_v1",
        "source": {
            "base": str(base_root),
            "dataset": str(args.source.resolve()),
            "reserve_max_distance_m": args.max_distance,
        },
        "episodes": combined_routes,
    }
    summary = {
        "schema_version": 1,
        "name": manifest["name"],
        "base_routes": len(base_routes),
        "reserve_routes": sum(len(values) for values in reserves.values()),
        "total_routes": len(combined_routes),
        "reserves": {
            split: dict(sorted(Counter(item.pattern for item in values).items()))
            for split, values in reserves.items()
        },
        "candidate_count": len(candidates),
        "geometry_rejected": dict(sorted(geometry_rejected.items())),
        "navmesh_rejected": dict(sorted(navmesh_rejected.items())),
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add balanced reserve routes to an AgentVLN candidate set"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scenes-root", type=Path, default=Path("data/scene_datasets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-pattern", type=int, default=5)
    parser.add_argument("--max-distance", type=float, default=6.0)
    parser.add_argument("--max-per-scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.per_pattern < 1 or args.max_per_scene < 1:
        raise ValueError("reserve and per-scene counts must be positive")
    print(json.dumps(extend(args), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

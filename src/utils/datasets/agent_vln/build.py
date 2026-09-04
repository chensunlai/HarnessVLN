from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


PATTERNS = ("low_turn", "one_turn", "two_turn")
STYLES = ("concise", "standard", "detailed")
SPLIT_RATIOS = (("debug", 0.6), ("dev", 0.2), ("test", 0.2))
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
TAG_PATTERNS = {
    "doorway": re.compile(r"\b(?:door|doorway|entrance|enter|exit)\w*\b", re.I),
    "room": re.compile(
        r"\b(?:room|hall|hallway|kitchen|bathroom|bedroom|office|lobby)\w*\b",
        re.I,
    ),
    "landmark_relative": re.compile(
        r"\b(?:past|around|beside|toward|towards|between|near|next to)\b", re.I
    ),
    "left": re.compile(r"\bleft\b", re.I),
    "right": re.compile(r"\bright\b", re.I),
    "endpoint": re.compile(r"\b(?:stop|wait|stand)\w*\b", re.I),
}


@dataclass(frozen=True)
class InstructionVariant:
    source_episode_id: str
    text: str
    tokens: tuple[int, ...]
    word_count: int
    style: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_episode_id": self.source_episode_id,
            "text": self.text,
            "tokens": list(self.tokens),
            "word_count": self.word_count,
            "style": self.style,
        }


@dataclass(frozen=True)
class Candidate:
    trajectory_id: str
    representative: Mapping[str, Any]
    variants: tuple[InstructionVariant, ...]
    pattern: str
    semantic_tags: tuple[str, ...]
    geometry: Mapping[str, Any]
    navmesh: Mapping[str, Any] | None = None

    @property
    def scene_id(self) -> str:
        return str(self.representative["scene_id"])


def _load_json(path: Path) -> Mapping[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping) or not isinstance(value.get("episodes"), list):
        raise ValueError(f"{path} is not an R2R-CE dataset document")
    return value


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def instruction_style(word_count: int) -> str:
    if word_count <= 14:
        return "concise"
    if word_count <= 25:
        return "standard"
    return "detailed"


def _instruction_variants(
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[InstructionVariant, ...]:
    values: list[InstructionVariant] = []
    seen: set[str] = set()
    for episode in episodes:
        instruction = episode.get("instruction")
        if not isinstance(instruction, Mapping):
            continue
        text = " ".join(str(instruction.get("instruction_text", "")).split())
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        tokens = instruction.get("instruction_tokens", ())
        word_count = len(_words(text))
        values.append(
            InstructionVariant(
                source_episode_id=str(episode.get("episode_id", "")),
                text=text,
                tokens=tuple(int(token) for token in tokens),
                word_count=word_count,
                style=instruction_style(word_count),
            )
        )
    return tuple(values)


def _turn_angles(path: Sequence[Sequence[float]]) -> list[float]:
    angles: list[float] = []
    for first, pivot, last in zip(path, path[1:], path[2:]):
        incoming = (pivot[0] - first[0], pivot[2] - first[2])
        outgoing = (last[0] - pivot[0], last[2] - pivot[2])
        norm_a = math.hypot(*incoming)
        norm_b = math.hypot(*outgoing)
        if min(norm_a, norm_b) <= 0.05:
            continue
        cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (
            norm_a * norm_b
        )
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return angles


def _initial_turn_angle(
    episode: Mapping[str, Any], path: Sequence[Sequence[float]]
) -> float:
    rotation = episode.get("start_rotation")
    if not isinstance(rotation, Sequence) or len(rotation) != 4:
        raise ValueError("start_rotation must be an [x, y, z, w] quaternion")
    x, y, z, w = (float(value) for value in rotation)
    # Habitat agents look down local -Z. Rotate that vector by the start quaternion.
    forward = (-2.0 * (x * z + y * w), -(1.0 - 2.0 * (x * x + y * y)))
    first_segment = (path[1][0] - path[0][0], path[1][2] - path[0][2])
    norm_forward = math.hypot(*forward)
    norm_segment = math.hypot(*first_segment)
    if min(norm_forward, norm_segment) <= 0.05:
        return 0.0
    cosine = (
        forward[0] * first_segment[0] + forward[1] * first_segment[1]
    ) / (norm_forward * norm_segment)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _route_geometry(episode: Mapping[str, Any]) -> dict[str, Any]:
    path = episode.get("reference_path")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError("reference_path must contain at least two points")
    points = [[float(value) for value in point] for point in path]
    turns = _turn_angles(points)
    vertical_span = max(point[1] for point in points) - min(point[1] for point in points)
    polyline = sum(math.dist(first, last) for first, last in zip(points, points[1:]))
    endpoint = math.dist(points[0], points[-1])
    source_distance = float(episode.get("info", {}).get("geodesic_distance", polyline))
    return {
        "source_geodesic_distance_m": round(source_distance, 6),
        "reference_polyline_m": round(polyline, 6),
        "endpoint_distance_m": round(endpoint, 6),
        "detour_ratio": round(source_distance / max(endpoint, 1e-6), 6),
        "viewpoint_count": len(points),
        "vertical_span_m": round(vertical_span, 6),
        "initial_turn_degrees": round(_initial_turn_angle(episode, points), 3),
        "turn_angles_degrees": [round(value, 3) for value in turns],
        "large_turn_count": sum(value >= 30.0 for value in turns),
        "max_turn_degrees": round(max(turns, default=0.0), 3),
    }


def _same_route(episodes: Sequence[Mapping[str, Any]]) -> None:
    first = episodes[0]
    keys = ("scene_id", "start_position", "start_rotation", "reference_path", "goals")
    for episode in episodes[1:]:
        if any(episode.get(key) != first.get(key) for key in keys):
            raise ValueError(f"trajectory {first.get('trajectory_id')} has inconsistent episodes")


def collect_candidates(
    document: Mapping[str, Any],
    *,
    max_distance_m: float = 5.5,
    max_vertical_span_m: float = 0.35,
    max_viewpoints: int = 6,
    max_large_turns: int = 2,
    max_turn_degrees: float = 135.0,
) -> tuple[list[Candidate], Counter[str]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in document["episodes"]:
        if not isinstance(raw, Mapping):
            raise ValueError("episode entries must be objects")
        trajectory_id = str(raw.get("trajectory_id", raw.get("episode_id", "")))
        groups[trajectory_id].append(raw)

    rejected: Counter[str] = Counter()
    candidates: list[Candidate] = []
    for trajectory_id, episodes in groups.items():
        _same_route(episodes)
        representative = episodes[0]
        variants = _instruction_variants(episodes)
        geometry = _route_geometry(representative)
        reason = None
        if geometry["source_geodesic_distance_m"] > max_distance_m:
            reason = "distance"
        elif geometry["viewpoint_count"] > max_viewpoints:
            reason = "viewpoints"
        elif geometry["vertical_span_m"] > max_vertical_span_m:
            reason = "vertical_span"
        elif geometry["large_turn_count"] > max_large_turns:
            reason = "large_turns"
        elif geometry["max_turn_degrees"] > max_turn_degrees:
            reason = "sharp_turn"
        elif len(variants) < 2:
            reason = "instruction_variants"
        if reason is not None:
            rejected[reason] += 1
            continue

        large_turns = int(geometry["large_turn_count"])
        pattern = PATTERNS[min(large_turns, len(PATTERNS) - 1)]
        combined_text = " ".join(variant.text for variant in variants)
        tags = tuple(
            name for name, regex in TAG_PATTERNS.items() if regex.search(combined_text)
        )
        candidates.append(
            Candidate(
                trajectory_id,
                representative,
                variants,
                pattern,
                tags,
                geometry,
            )
        )
    return candidates, rejected


def validate_navmeshes(
    candidates: Sequence[Candidate], scenes_root: Path, *, max_distance_m: float
) -> tuple[list[Candidate], Counter[str]]:
    try:
        import habitat_sim
        import numpy as np
    except ImportError as error:
        raise RuntimeError("--validate-navmesh requires habitat_sim and numpy") from error

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.scene_id].append(candidate)
    accepted: list[Candidate] = []
    rejected: Counter[str] = Counter()
    for scene_id in sorted(grouped):
        navmesh_path = (scenes_root / scene_id).with_suffix(".navmesh")
        pathfinder = habitat_sim.PathFinder()
        if not navmesh_path.is_file() or not pathfinder.load_nav_mesh(str(navmesh_path)):
            rejected["missing_navmesh"] += len(grouped[scene_id])
            continue
        for candidate in grouped[scene_id]:
            points = [
                np.asarray(point, dtype=np.float32)
                for point in candidate.representative["reference_path"]
            ]
            if not all(pathfinder.is_navigable(point) for point in points):
                rejected["non_navigable_waypoint"] += 1
                continue
            shortest = habitat_sim.ShortestPath()
            shortest.requested_start = points[0]
            shortest.requested_end = points[-1]
            if not pathfinder.find_path(shortest) or not math.isfinite(
                shortest.geodesic_distance
            ):
                rejected["no_navmesh_path"] += 1
                continue
            if shortest.geodesic_distance > max_distance_m + 1.0:
                rejected["navmesh_distance"] += 1
                continue
            validation = {
                "status": "passed",
                "navmesh_path": str(navmesh_path),
                "all_reference_points_navigable": True,
                "geodesic_distance_m": round(float(shortest.geodesic_distance), 6),
            }
            accepted.append(replace(candidate, navmesh=validation))
    return accepted, rejected


def _stable_key(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest()


def split_sizes(count: int) -> dict[str, int]:
    if count < 3:
        raise ValueError("count must be at least 3")
    debug = round(count * SPLIT_RATIOS[0][1])
    dev = round(count * SPLIT_RATIOS[1][1])
    return {"debug": debug, "dev": dev, "test": count - debug - dev}


def partition_scenes(
    candidates: Sequence[Candidate], sizes: Mapping[str, int], seed: int
) -> dict[str, set[str]]:
    scenes = sorted(
        {candidate.scene_id for candidate in candidates},
        key=lambda scene: _stable_key(seed, "scene", scene),
    )
    if len(scenes) < len(sizes):
        raise ValueError("not enough scenes for scene-disjoint splits")
    boundaries: list[int] = []
    cumulative = 0.0
    total = sum(sizes.values())
    for name, _ in SPLIT_RATIOS[:-1]:
        cumulative += sizes[name] / total
        boundaries.append(round(len(scenes) * cumulative))
    parts = [
        scenes[: boundaries[0]],
        scenes[boundaries[0] : boundaries[1]],
        scenes[boundaries[1] :],
    ]
    return {name: set(part) for (name, _), part in zip(SPLIT_RATIOS, parts)}


def _target_pairs(count: int) -> list[tuple[str, str]]:
    return [
        (PATTERNS[index % len(PATTERNS)], STYLES[(index // 3 + index) % len(STYLES)])
        for index in range(count)
    ]


def select_split(
    candidates: Sequence[Candidate],
    *,
    count: int,
    seed: int,
    split: str,
    max_per_scene: int,
    max_instruction_words: int,
) -> list[tuple[Candidate, InstructionVariant]]:
    remaining = list(candidates)
    selected: list[tuple[Candidate, InstructionVariant]] = []
    scene_counts: Counter[str] = Counter()
    for index, (pattern, style) in enumerate(_target_pairs(count)):
        eligible = [
            candidate
            for candidate in remaining
            if candidate.pattern == pattern
            and style
            in {
                variant.style
                for variant in candidate.variants
                if variant.word_count <= max_instruction_words
            }
            and scene_counts[candidate.scene_id] < max_per_scene
        ]
        if not eligible:
            raise ValueError(
                f"split {split!r} cannot fill pattern={pattern}, style={style}; "
                "relax bounds or max_per_scene"
            )
        candidate = min(
            eligible,
            key=lambda value: (
                scene_counts[value.scene_id],
                _stable_key(seed, f"{split}:{index}", value.trajectory_id),
            ),
        )
        variants = [
            variant
            for variant in candidate.variants
            if variant.style == style and variant.word_count <= max_instruction_words
        ]
        primary = min(
            variants,
            key=lambda value: _stable_key(
                seed, f"{split}:instruction", value.source_episode_id
            ),
        )
        selected.append((candidate, primary))
        scene_counts[candidate.scene_id] += 1
        remaining.remove(candidate)
    return selected


def _native_episode(
    index: int, split: str, candidate: Candidate, primary: InstructionVariant
) -> dict[str, Any]:
    source = candidate.representative
    goals = json.loads(json.dumps(source.get("goals", [])))
    for goal in goals:
        if isinstance(goal, dict):
            goal["radius"] = 1.0
    info = dict(source.get("info", {}))
    info.update(
        {
            "agent_vln_split": split,
            "route_pattern": candidate.pattern,
            "semantic_tags": list(candidate.semantic_tags),
            "source_episode_id": primary.source_episode_id,
            "source_trajectory_id": candidate.trajectory_id,
        }
    )
    return {
        "episode_id": f"agent_vln:{split}:{index:04d}",
        "trajectory_id": source.get("trajectory_id", candidate.trajectory_id),
        "scene_id": source["scene_id"],
        "start_position": source["start_position"],
        "start_rotation": source["start_rotation"],
        "info": info,
        "goals": goals,
        "instruction": {
            "instruction_text": primary.text,
            "instruction_tokens": list(primary.tokens),
        },
        "reference_path": source["reference_path"],
    }


def _manifest_episode(
    index: int, split: str, candidate: Candidate, primary: InstructionVariant
) -> dict[str, Any]:
    return {
        "episode_id": f"agent_vln:{split}:{index:04d}",
        "split": split,
        "scene_id": candidate.scene_id,
        "source": {
            "dataset": "R2R-CE",
            "split": "train",
            "trajectory_id": candidate.trajectory_id,
            "primary_episode_id": primary.source_episode_id,
        },
        "route_pattern": candidate.pattern,
        "semantic_tags": list(candidate.semantic_tags),
        "geometry": dict(candidate.geometry),
        "navmesh_validation": dict(candidate.navmesh or {"status": "not_run"}),
        "visual_validation": "pending",
        "primary_instruction": primary.as_dict(),
        "instruction_variants": [variant.as_dict() for variant in candidate.variants],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(records),
        "scenes": len({str(record["scene_id"]) for record in records}),
        "route_patterns": dict(
            sorted(Counter(record["route_pattern"] for record in records).items())
        ),
        "instruction_styles": dict(
            sorted(Counter(record["primary_instruction"]["style"] for record in records).items())
        ),
        "semantic_tags": dict(
            sorted(Counter(tag for record in records for tag in record["semantic_tags"]).items())
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    document = _load_json(source)
    candidates, geometry_rejected = collect_candidates(
        document,
        max_distance_m=args.max_distance,
        max_vertical_span_m=args.max_vertical_span,
        max_viewpoints=args.max_viewpoints,
        max_large_turns=args.max_large_turns,
        max_turn_degrees=args.max_turn,
    )
    before_navmesh = len(candidates)
    navmesh_rejected: Counter[str] = Counter()
    if args.validate_navmesh:
        candidates, navmesh_rejected = validate_navmeshes(
            candidates, args.scenes_root.resolve(), max_distance_m=args.max_distance
        )

    sizes = split_sizes(args.count)
    scene_splits = partition_scenes(candidates, sizes, args.seed)
    output = args.output.resolve()
    manifests: list[dict[str, Any]] = []
    for split, size in sizes.items():
        pool = [
            candidate
            for candidate in candidates
            if candidate.scene_id in scene_splits[split]
        ]
        selected = select_split(
            pool,
            count=size,
            seed=args.seed,
            split=split,
            max_per_scene=args.max_per_scene,
            max_instruction_words=args.max_instruction_words,
        )
        native = [
            _native_episode(index, split, candidate, primary)
            for index, (candidate, primary) in enumerate(selected)
        ]
        records = [
            _manifest_episode(index, split, candidate, primary)
            for index, (candidate, primary) in enumerate(selected)
        ]
        manifests.extend(records)
        _write_gzip_json(
            output / split / f"{split}.json.gz",
            {"instruction_vocab": document.get("instruction_vocab", {}), "episodes": native},
        )

    scene_sets = {
        split: {record["scene_id"] for record in manifests if record["split"] == split}
        for split in sizes
    }
    overlap = {
        f"{left}:{right}": sorted(scene_sets[left] & scene_sets[right])
        for index, left in enumerate(sizes)
        for right in list(sizes)[index + 1 :]
    }
    manifest = {
        "schema_version": 1,
        "name": "agent_vln_r2r_local_v1",
        "source": str(source),
        "seed": args.seed,
        "episodes": manifests,
    }
    summary = {
        "schema_version": 1,
        "name": manifest["name"],
        "source": {
            "path": str(source),
            "sha256": _file_sha256(source),
            "split": "train",
            "routes": len(
                {
                    str(episode.get("trajectory_id", episode.get("episode_id")))
                    for episode in document["episodes"]
                }
            ),
        },
        "selection": {
            "requested": args.count,
            "geometry_candidates": before_navmesh,
            "navmesh_candidates": len(candidates),
            "geometry_rejected": dict(sorted(geometry_rejected.items())),
            "navmesh_rejected": dict(sorted(navmesh_rejected.items())),
            "max_distance_m": args.max_distance,
            "max_vertical_span_m": args.max_vertical_span,
            "max_viewpoints": args.max_viewpoints,
            "max_large_turns": args.max_large_turns,
            "max_turn_degrees": args.max_turn,
            "max_per_scene": args.max_per_scene,
            "max_instruction_words": args.max_instruction_words,
        },
        "splits": {
            split: _distribution([record for record in manifests if record["split"] == split])
            for split in sizes
        },
        "checks": {
            "episode_count": len(manifests),
            "unique_trajectories": len(
                {record["source"]["trajectory_id"] for record in manifests}
            ),
            "scene_overlap": overlap,
            "scene_disjoint": not any(overlap.values()),
            "all_navmesh_validated": all(
                record["navmesh_validation"]["status"] == "passed" for record in manifests
            ) if args.validate_navmesh else False,
            "visual_validation": "pending",
        },
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local AgentVLN set from R2R-CE")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scenes-root", type=Path, default=Path("data/scene_datasets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-distance", type=float, default=5.5)
    parser.add_argument("--max-vertical-span", type=float, default=0.35)
    parser.add_argument("--max-viewpoints", type=int, default=6)
    parser.add_argument("--max-large-turns", type=int, default=2)
    parser.add_argument("--max-turn", type=float, default=135.0)
    parser.add_argument("--max-per-scene", type=int, default=4)
    parser.add_argument("--max-instruction-words", type=int, default=40)
    parser.add_argument("--validate-navmesh", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image, ImageDraw
from tqdm import tqdm


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


def _load_r2r(path: Path) -> Mapping[str, Any]:
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


def _initial_alignment(episode: Mapping[str, Any]) -> dict[str, Any]:
    path = episode.get("reference_path")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError("reference_path must contain at least two points")
    rotation = episode.get("start_rotation")
    if not isinstance(rotation, Sequence) or len(rotation) != 4:
        raise ValueError("start_rotation must be an [x, y, z, w] quaternion")
    x, y, z, w = (float(value) for value in rotation)
    forward = (-2.0 * (x * z + y * w), -(1.0 - 2.0 * (x * x + y * y)))
    segment = (path[1][0] - path[0][0], path[1][2] - path[0][2])
    angle = _initial_turn_angle(episode, path)
    cross = forward[0] * segment[1] - forward[1] * segment[0]
    direction = "straight" if angle < 15.0 else "right" if cross > 0 else "left"
    return {"direction": direction, "degrees": round(angle, 3)}


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
    document = _load_r2r(source)
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
    source = _load_r2r(args.source.resolve())
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




SENSOR_UUID = "route_rgb"


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def safe_route_id(route_id: str) -> str:
    return route_id.replace(":", "_").replace("/", "_")


def uniform_indices(size: int, limit: int) -> list[int]:
    if size < 1 or limit < 1:
        raise ValueError("size and limit must be positive")
    if size <= limit:
        return list(range(size))
    return [round(index * (size - 1) / (limit - 1)) for index in range(limit)]


def _direction_rotation(first: Sequence[float], last: Sequence[float]) -> Any:
    import numpy as np
    from habitat_sim.utils.common import quat_from_angle_axis

    dx = float(last[0]) - float(first[0])
    dz = float(last[2]) - float(first[2])
    if math.hypot(dx, dz) <= 0.05:
        raise ValueError("cannot orient a route frame along a zero-length segment")
    yaw = math.atan2(-dx, -dz)
    return quat_from_angle_axis(yaw, np.asarray([0.0, 1.0, 0.0]))


def frame_rotation(episode: Mapping[str, Any], path_index: int) -> Any:
    from habitat_sim.utils.common import quat_from_coeffs

    path = episode["reference_path"]
    if path_index == 0:
        return quat_from_coeffs(episode["start_rotation"])
    if path_index < len(path) - 1:
        return _direction_rotation(path[path_index], path[path_index + 1])
    return _direction_rotation(path[path_index - 1], path[path_index])


def quaternion_coefficients(rotation: Any) -> list[float]:
    return [
        float(rotation.imag[0]),
        float(rotation.imag[1]),
        float(rotation.imag[2]),
        float(rotation.real),
    ]


def create_simulator(
    scene_path: Path,
    *,
    gpu_device_id: int,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
) -> Any:
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene_path)
    simulator_config.gpu_device_id = gpu_device_id
    simulator_config.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = SENSOR_UUID
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [height, width]
    sensor.position = [0.0, sensor_height, 0.0]
    sensor.hfov = hfov

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [sensor]
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent_config])
    )


def _render_frame(
    simulator: Any, position: Sequence[float], rotation: Any
) -> Image.Image:
    import habitat_sim
    import numpy as np

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = rotation
    simulator.get_agent(0).set_state(state, reset_sensors=True)
    pixels = np.asarray(simulator.get_sensor_observations()[SENSOR_UUID])
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError(f"unexpected RGB observation shape: {pixels.shape}")
    rgb = np.ascontiguousarray(pixels[:, :, :3].astype(np.uint8))
    if float(rgb.std()) < 1.0:
        raise ValueError("rendered route frame is blank or nearly constant")
    return Image.fromarray(rgb, mode="RGB")


def _contact_sheet(images: Sequence[Image.Image], labels: Sequence[str]) -> Image.Image:
    columns = 2
    label_height = 28
    rows = math.ceil(len(images) / columns)
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        draw.text((x + 8, y + 7), label, fill="black")
        sheet.paste(image, (x, y + label_height))
    return sheet


def render_route(
    simulator: Any,
    route: Mapping[str, Any],
    episode: Mapping[str, Any],
    output_root: Path,
    *,
    max_frames: int,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
    overwrite: bool,
) -> Mapping[str, Any]:
    route_id = str(route["episode_id"])
    split = str(route["split"])
    directory = output_root / "images" / split / safe_route_id(route_id)
    metadata_path = directory / "frames.json"
    if metadata_path.is_file() and not overwrite:
        metadata = _read_json(metadata_path)
        frame_paths = [output_root / value["path"] for value in metadata["frames"]]
        if frame_paths and all(path.is_file() for path in frame_paths):
            return metadata

    path = episode.get("reference_path")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError(f"route {route_id} has no usable reference_path")
    indices = uniform_indices(len(path), max_frames)
    directory.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    labels: list[str] = []
    for frame_index, path_index in enumerate(indices):
        rotation = frame_rotation(episode, path_index)
        image = _render_frame(simulator, path[path_index], rotation)
        filename = f"frame_{frame_index:02d}.jpg"
        image_path = directory / filename
        image.save(image_path, format="JPEG", quality=90, optimize=True)
        role = (
            "start"
            if path_index == 0
            else "goal"
            if path_index == len(path) - 1
            else "route"
        )
        relative_path = image_path.relative_to(output_root)
        frames.append(
            {
                "index": frame_index,
                "path_index": path_index,
                "role": role,
                "path": str(relative_path),
                "position": [float(value) for value in path[path_index]],
                "rotation": quaternion_coefficients(rotation),
                "sha256": _file_sha256(image_path),
            }
        )
        images.append(image)
        labels.append(f"{frame_index + 1}/{len(indices)}  {role}")

    sheet_path = directory / "contact_sheet.jpg"
    sheet = _contact_sheet(images, labels)
    sheet.save(sheet_path, format="JPEG", quality=90, optimize=True)
    metadata = {
        "schema_version": 1,
        "route_id": route_id,
        "scene_id": route["scene_id"],
        "render": {
            "width": width,
            "height": height,
            "hfov_degrees": hfov,
            "sensor_height_m": sensor_height,
            "orientation": (
                "original episode heading at start; outgoing route direction at "
                "intermediate points; arrival direction at goal"
            ),
        },
        "frames": frames,
        "contact_sheet": str(sheet_path.relative_to(output_root)),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def cached_route_frames(
    route: Mapping[str, Any], output_root: Path, args: argparse.Namespace
) -> Mapping[str, Any] | None:
    path = (
        output_root
        / "images"
        / str(route["split"])
        / safe_route_id(str(route["episode_id"]))
        / "frames.json"
    )
    if not path.is_file() or args.overwrite:
        return None
    try:
        metadata = _read_json(path)
        render = metadata["render"]
        if (
            metadata.get("route_id") != route["episode_id"]
            or metadata.get("scene_id") != route["scene_id"]
            or render.get("width") != args.width
            or render.get("height") != args.height
            or float(render.get("hfov_degrees")) != args.hfov
            or float(render.get("sensor_height_m")) != args.sensor_height
        ):
            return None
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            return None
        if any(not (output_root / frame["path"]).is_file() for frame in frames):
            return None
        return metadata
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_routes(
    dataset_root: Path,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    import gzip

    manifest = _read_json(dataset_root / "manifest.json")
    routes = manifest.get("episodes")
    if not isinstance(routes, list):
        raise ValueError("input manifest must contain an episodes list")
    episodes: dict[str, Mapping[str, Any]] = {}
    for split in sorted({str(route["split"]) for route in routes}):
        path = dataset_root / split / f"{split}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
        episodes.update(
            (str(episode["episode_id"]), episode) for episode in document["episodes"]
        )
    missing = [
        route["episode_id"]
        for route in routes
        if route["episode_id"] not in episodes
    ]
    if missing:
        raise ValueError(
            f"native split files are missing {len(missing)} manifest routes"
        )
    return [(route, episodes[str(route["episode_id"])]) for route in routes]


def render_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.input.resolve()
    output_root = args.output.resolve()
    routes = load_routes(dataset_root)
    if args.limit is not None:
        routes = routes[: args.limit]
    cached = 0
    pending = []
    for route, episode in routes:
        if cached_route_frames(route, output_root, args) is not None:
            cached += 1
        else:
            pending.append((route, episode))
    by_scene: dict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    for route, episode in pending:
        by_scene[str(route["scene_id"])].append((route, episode))
    scenes = [
        item
        for index, item in enumerate(sorted(by_scene.items()))
        if index % args.num_shards == args.shard_index
    ]

    rendered = 0
    shard_routes = sum(len(scene_routes) for _, scene_routes in scenes)
    progress = tqdm(
        total=shard_routes,
        desc=f"render shard {args.shard_index + 1}/{args.num_shards}",
        unit="route",
        dynamic_ncols=True,
    )
    try:
        for scene_id, scene_routes in scenes:
            scene_path = args.scenes_root.resolve() / scene_id
            if not scene_path.is_file():
                raise FileNotFoundError(f"scene does not exist: {scene_path}")
            with _quiet_native_output(args.quiet_native_logs):
                simulator = create_simulator(
                    scene_path,
                    gpu_device_id=args.gpu_device_id,
                    width=args.width,
                    height=args.height,
                    hfov=args.hfov,
                    sensor_height=args.sensor_height,
                )
            try:
                for route, episode in scene_routes:
                    try:
                        render_route(
                            simulator,
                            route,
                            episode,
                            output_root,
                            max_frames=args.max_frames,
                            width=args.width,
                            height=args.height,
                            hfov=args.hfov,
                            sensor_height=args.sensor_height,
                            overwrite=args.overwrite,
                        )
                    except Exception as error:
                        progress.write(
                            f"ERROR render {route['episode_id']}: "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                        )
                        raise
                    rendered += 1
                    progress.update()
            finally:
                with _quiet_native_output(args.quiet_native_logs):
                    simulator.close()
    finally:
        progress.close()
    return {
        "routes": rendered,
        "cached": cached,
        "scenes": len(scenes),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "output": str(output_root),
    }


@contextlib.contextmanager
def _quiet_native_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    null_fd = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(null_fd)




PROMPT_VERSION = "agent_vln_rewrite_v3"
STYLE_ORDER = ("concise", "natural", "landmark_rich")
STYLE_LIMITS = {
    "concise": (6, 22),
    "natural": (12, 36),
    "landmark_rich": (20, 52),
}
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")
FORBIDDEN_MEDIA_REFERENCE = re.compile(
    r"\b(?:dataset|annotation|waypoint|screenshot)\b|"
    r"\b(?:shown|visible|seen|depicted)\s+in\s+(?:the\s+)?"
    r"(?:(?:first|last|current|previous|next|route)\s+)?"
    r"(?:image|frame|photo|picture)\b|"
    r"\b(?:route|path|object|door|room)\s+(?:shown|visible|seen|depicted)\s+in\s+"
    r"(?:the\s+)?(?:(?:first|last|current|previous|next|route)\s+)?"
    r"(?:image|frame|photo|picture)\b|"
    r"\b(?:image|frame|photo|picture)\s+(?:shows|depicts|indicates)\b",
    re.I,
)

SYSTEM_INSTRUCTIONS = """You curate grounded English instructions for indoor
vision-and-language navigation.

The input contains several human instructions for one ground-truth route followed by
chronologically ordered RGB samples from that route. The first image uses the agent's
true starting heading. Intermediate images face the next route segment, and the final
image faces the arrival direction. Black regions can be missing Matterport scan data;
ignore them.

The human instructions propose the route, action order, and endpoint. The supplied
initial_alignment is computed from the annotated start pose and first route segment;
it is authoritative. Include its left or right turn when it is not straight, even if
the first RGB sample alone makes several openings plausible. The ordered RGB samples
determine which room and landmark wording is visually executable. Retain the shared
route geometry but omit a room, object, or relation that the samples do not support.
Prefer a supported transition or landmark in its place. Never instruct the agent to
search for an unseen landmark merely because one annotation names it.

Resolve viewpoint semantics carefully. A room named in an annotation may be the room
the agent is already leaving, not another room it must enter. Describe it that way when
the first sample and route order support that reading. Every generated clause must move
the agent forward along the sampled route rather than into a plausible side opening.
When initial_alignment requires a turn, never call the competing initial forward view
"ahead" or direct the agent into it.
Do not invent an object, doorway, turn, room transition, or destination. Write commands
executable by an agent starting at the first sample. Never mention images, frames,
coordinates, waypoints, datasets, or annotations.

Return exactly three semantically equivalent instructions with distinct styles:
- concise: 6-22 words, direct imperative, only essential turns and endpoint;
- natural: 12-36 words, fluent directions a person would naturally give;
- landmark_rich: 20-52 words, ordered steps with only clearly supported landmarks.

Assess the generated instructions, rather than every detail in the source wording. Use
grounded when all facts retained in the generated instructions are supported. Use
partially_grounded only when an essential route transition remains visually ambiguous,
and conflict only when no faithful executable rewrite can resolve a material mismatch.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "route_check": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["grounded", "partially_grounded", "conflict"],
                },
                "notes": {"type": "string", "minLength": 1, "maxLength": 300},
                "verified_landmarks": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 80},
                    "maxItems": 8,
                },
            },
            "required": ["status", "notes", "verified_landmarks"],
            "additionalProperties": False,
        },
        "instructions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "style": {"type": "string", "enum": list(STYLE_ORDER)},
                    "text": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["style", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["route_check", "instructions"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class RewriteJob:
    route: Mapping[str, Any]
    episode: Mapping[str, Any]
    frame_manifest: Mapping[str, Any]
    fingerprint: str

    @property
    def route_id(self) -> str:
        return str(self.route["episode_id"])

    @property
    def split(self) -> str:
        return str(self.route["split"])


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def validate_generation(value: Mapping[str, Any]) -> dict[str, Any]:
    route_check = value.get("route_check")
    if not isinstance(route_check, Mapping):
        raise ValueError("route_check must be an object")
    if route_check.get("status") not in {"grounded", "partially_grounded", "conflict"}:
        raise ValueError("route_check.status is invalid")
    notes = route_check.get("notes")
    landmarks = route_check.get("verified_landmarks")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("route_check.notes must be non-empty")
    if not isinstance(landmarks, list) or not all(
        isinstance(item, str) and item.strip() for item in landmarks
    ):
        raise ValueError("verified_landmarks must be a string list")

    instructions = value.get("instructions")
    if not isinstance(instructions, list) or len(instructions) != 3:
        raise ValueError("exactly three instructions are required")
    by_style: dict[str, dict[str, Any]] = {}
    normalized: set[str] = set()
    for item in instructions:
        if not isinstance(item, Mapping):
            raise ValueError("instruction entries must be objects")
        style = item.get("style")
        text = item.get("text")
        if style not in STYLE_ORDER or style in by_style:
            raise ValueError("instruction styles must be unique and recognized")
        if not isinstance(text, str) or not (clean := _normalize_text(text)):
            raise ValueError(f"{style} instruction must be non-empty")
        if FORBIDDEN_MEDIA_REFERENCE.search(clean):
            raise ValueError(f"{style} instruction refers to dataset media")
        minimum, maximum = STYLE_LIMITS[str(style)]
        count = _word_count(clean)
        if not minimum <= count <= maximum:
            raise ValueError(
                f"{style} instruction has {count} words; expected {minimum}-{maximum}"
            )
        key = re.sub(r"\W+", " ", clean.casefold()).strip()
        if key in normalized:
            raise ValueError("generated instructions are not distinct")
        normalized.add(key)
        by_style[str(style)] = {
            "style": str(style),
            "text": clean,
            "word_count": count,
        }
    if set(by_style) != set(STYLE_ORDER):
        raise ValueError("one instruction per required style is required")
    return {
        "route_check": {
            "status": str(route_check["status"]),
            "notes": _normalize_text(str(notes)),
            "verified_landmarks": [_normalize_text(item) for item in landmarks],
        },
        "instructions": [by_style[style] for style in STYLE_ORDER],
    }


def validate_route_generation(
    value: Mapping[str, Any], episode: Mapping[str, Any]
) -> dict[str, Any]:
    result = validate_generation(value)
    direction = _initial_alignment(episode)["direction"]
    if direction != "straight" and any(
        not re.search(rf"\b{direction}\b", item["text"], re.I)
        for item in result["instructions"]
    ):
        raise ValueError(f"every instruction must retain the initial {direction} turn")
    return result


def _job_fingerprint(
    route: Mapping[str, Any], frame_manifest: Mapping[str, Any]
) -> str:
    source = {
        "route_id": route["episode_id"],
        "instructions": [item["text"] for item in route["instruction_variants"]],
        "frames": [item["sha256"] for item in frame_manifest["frames"]],
        "prompt_version": PROMPT_VERSION,
    }
    payload = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_jobs(
    input_root: Path, output_root: Path, *, limit: int | None = None
) -> list[RewriteJob]:
    jobs: list[RewriteJob] = []
    routes = load_routes(input_root)
    if limit is not None:
        routes = routes[:limit]
    for route, episode in routes:
        frame_path = (
            output_root
            / "images"
            / str(route["split"])
            / safe_route_id(str(route["episode_id"]))
            / "frames.json"
        )
        if not frame_path.is_file():
            raise FileNotFoundError(
                f"missing route images for {route['episode_id']}: {frame_path}"
            )
        frame_manifest = _read_json(frame_path)
        frames = frame_manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"route {route['episode_id']} has no rendered frames")
        missing = [
            item["path"]
            for item in frames
            if not (output_root / item["path"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"route {route['episode_id']} is missing {len(missing)} frame files"
            )
        jobs.append(
            RewriteJob(
                route,
                episode,
                frame_manifest,
                _job_fingerprint(route, frame_manifest),
            )
        )
    return jobs


def _generation_path(output_root: Path, job: RewriteJob) -> Path:
    return (
        output_root
        / "generations"
        / job.split
        / f"{safe_route_id(job.route_id)}.json"
    )


def _cached_generation(
    path: Path, *, fingerprint: str, model: str, reasoning_effort: str
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = _read_json(path)
        if (
            record.get("fingerprint") != fingerprint
            or record.get("model") != model
            or record.get("reasoning_effort") != reasoning_effort
            or record.get("prompt_version") != PROMPT_VERSION
        ):
            return None
        validate_generation(record["result"])
        return record
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _request_input(
    job: RewriteJob,
    output_root: Path,
    *,
    image_detail: str,
    feedback: str | None,
) -> list[dict[str, Any]]:
    route = job.route
    source = {
        "route_id": job.route_id,
        "approximate_distance_m": route["geometry"]["source_geodesic_distance_m"],
        "initial_alignment": _initial_alignment(job.episode),
        "route_pattern": route["route_pattern"],
        "human_instructions": [
            item["text"] for item in route["instruction_variants"]
        ],
    }
    text = (
        "Rewrite this single route according to the developer instructions. "
        "The RGB samples follow in chronological order.\n"
        + json.dumps(source, ensure_ascii=False, indent=2)
    )
    if feedback:
        text += f"\nThe previous output failed validation: {feedback}. Correct it."
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    frames = job.frame_manifest["frames"]
    for index, frame in enumerate(frames):
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Ordered route sample {index + 1}/{len(frames)} "
                    f"({frame['role']})."
                ),
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": _image_data_url(output_root / frame["path"]),
                "detail": image_detail,
            }
        )
    return [{"role": "user", "content": content}]


def _usage_dict(response: Any) -> Mapping[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "output_tokens_details", None)
    return {
        name: int(getattr(usage, name, 0) or 0)
        for name in ("input_tokens", "output_tokens", "total_tokens")
    } | {"reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0)}


async def _generate_one(
    client: Any,
    job: RewriteJob,
    output_root: Path,
    *,
    model: str,
    reasoning_effort: str,
    image_detail: str,
    retries: int,
    retry_backoff_s: float,
) -> Mapping[str, Any]:
    path = _generation_path(output_root, job)
    cached = _cached_generation(
        path,
        fingerprint=job.fingerprint,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    if cached is not None:
        try:
            validate_route_generation(cached["result"], job.episode)
            return cached
        except (KeyError, TypeError, ValueError):
            pass

    feedback: str | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=_request_input(
                    job,
                    output_root,
                    image_detail=image_detail,
                    feedback=feedback,
                ),
                reasoning={"effort": reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "agent_vln_instruction_variants",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    }
                },
                max_output_tokens=4096,
                store=False,
            )
            if getattr(response, "status", None) != "completed":
                raise RuntimeError(
                    f"response status={getattr(response, 'status', None)!r}; "
                    f"details={getattr(response, 'incomplete_details', None)!r}"
                )
            output_text = getattr(response, "output_text", "")
            if not output_text:
                raise ValueError("response has no output_text")
            result = validate_route_generation(json.loads(output_text), job.episode)
            record = {
                "schema_version": 1,
                "route_id": job.route_id,
                "fingerprint": job.fingerprint,
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "response_id": getattr(response, "id", None),
                "usage": _usage_dict(response),
                "result": result,
            }
            _write_json(path, record)
            return record
        except Exception as error:
            feedback = f"{type(error).__name__}: {error}"
            print(
                f"ERROR rewrite {job.route_id} attempt {attempt + 1}/{retries + 1}: "
                f"{feedback}",
                file=sys.stderr,
                flush=True,
            )
            if attempt >= retries:
                raise
            await asyncio.sleep(retry_backoff_s * (2**attempt))
    raise AssertionError("unreachable")


async def generate_all(
    jobs: Sequence[RewriteJob], output_root: Path, args: argparse.Namespace
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set in the process environment")
    client = AsyncOpenAI(timeout=args.api_timeout, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: dict[str, Mapping[str, Any]] = {}
    failures: list[dict[str, str]] = []

    async def run(job: RewriteJob) -> tuple[RewriteJob, Mapping[str, Any] | Exception]:
        async with semaphore:
            try:
                value = await _generate_one(
                    client,
                    job,
                    output_root,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    image_detail=args.image_detail,
                    retries=args.retries,
                    retry_backoff_s=args.retry_backoff,
                )
                return job, value
            except Exception as error:
                return job, error

    progress = tqdm(
        total=len(jobs), desc="rewrite routes", unit="route", dynamic_ncols=True
    )
    try:
        tasks = [asyncio.create_task(run(job)) for job in jobs]
        for task in asyncio.as_completed(tasks):
            job, value = await task
            if isinstance(value, Exception):
                failure = {
                    "route_id": job.route_id,
                    "type": type(value).__name__,
                    "message": str(value),
                }
                failures.append(failure)
                progress.write(
                    f"FAILED rewrite {job.route_id}: {failure['type']}: "
                    f"{failure['message']}",
                    file=sys.stderr,
                )
            else:
                results[job.route_id] = value
            progress.update()
            progress.set_postfix(ok=len(results), failed=len(failures))
    finally:
        progress.close()
        await client.close()
    return results, failures


def _tokenize(text: str, vocabulary: Mapping[str, Any]) -> list[int]:
    word_to_index = vocabulary.get("word2idx_dict", {})
    unknown = int(vocabulary.get("UNK_INDEX", 1))
    sentence = text.lower().replace("'s", " 's").replace(",", "").replace("?", "")
    tokens = [
        token.strip()
        for token in SENTENCE_SPLIT_REGEX.split(sentence)
        if token.strip()
    ]
    return [int(word_to_index.get(token, unknown)) for token in tokens]


def _source_documents(
    input_root: Path, splits: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    documents: dict[str, Mapping[str, Any]] = {}
    for split in splits:
        path = input_root / split / f"{split}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError(f"source split {split} must be an object")
        documents[split] = value
    return documents


def select_final_jobs(
    jobs: Sequence[RewriteJob],
    generations: Mapping[str, Mapping[str, Any]],
    *,
    count: int,
    min_distance: float = 0.0,
) -> tuple[list[RewriteJob], Mapping[str, Any]]:
    sizes = split_sizes(count)
    selected: list[RewriteJob] = []
    selected_ids: set[str] = set()
    required: dict[str, dict[str, int]] = {}
    for split, size in sizes.items():
        pattern_counts = {
            pattern: sum(
                PATTERNS[index % len(PATTERNS)] == pattern for index in range(size)
            )
            for pattern in PATTERNS
        }
        required[split] = pattern_counts
        for pattern, target in pattern_counts.items():
            eligible = [
                job
                for job in jobs
                if job.split == split
                and job.route["route_pattern"] == pattern
                and float(
                    job.route.get("geometry", {}).get(
                        "source_geodesic_distance_m", 0.0
                    )
                )
                >= min_distance
                and generations[job.route_id]["result"]["route_check"]["status"]
                != "conflict"
            ]
            if len(eligible) < target:
                raise ValueError(
                    f"only {len(eligible)} eligible routes for split={split}, "
                    f"pattern={pattern}, minimum_distance={min_distance}; "
                    f"need {target}"
                )
            for job in eligible[:target]:
                selected.append(job)
                selected_ids.add(job.route_id)

    conflicts = [
        job.route_id
        for job in jobs
        if generations[job.route_id]["result"]["route_check"]["status"] == "conflict"
    ]
    curation = {
        "policy": (
            "exclude generator-reviewed conflicts and routes below the minimum "
            "distance, then preserve split/pattern quotas"
        ),
        "minimum_geodesic_distance_m": min_distance,
        "candidate_routes": len(jobs),
        "selected_routes": len(selected),
        "required_patterns": required,
        "excluded_conflicts": conflicts,
        "unused_non_conflict": [
            job.route_id
            for job in jobs
            if job.route_id not in selected_ids and job.route_id not in conflicts
        ],
    }
    return selected, curation


def materialize(
    jobs: Sequence[RewriteJob],
    generations: Mapping[str, Mapping[str, Any]],
    *,
    input_root: Path,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    curation: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if missing := [job.route_id for job in jobs if job.route_id not in generations]:
        raise ValueError(
            f"cannot materialize: {len(missing)} routes have no generation"
        )
    splits = sorted({job.split for job in jobs})
    source_documents = _source_documents(input_root, splits)
    routes: list[dict[str, Any]] = []
    expanded: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}

    for job in jobs:
        generation = generations[job.route_id]
        result = validate_generation(generation["result"])
        route_record = dict(job.route)
        route_record["route_images"] = job.frame_manifest
        route_record["visual_validation"] = {
            "method": "generator_review",
            **result["route_check"],
        }
        route_record["generated_instructions"] = result["instructions"]
        route_record["generation"] = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "response_id": generation.get("response_id"),
            "usage": generation.get("usage", {}),
        }
        routes.append(route_record)

        vocabulary = source_documents[job.split]["instruction_vocab"]
        for instruction in result["instructions"]:
            episode = json.loads(json.dumps(job.episode))
            style = instruction["style"]
            episode["episode_id"] = f"{job.route_id}:{style}"
            episode["instruction"] = {
                "instruction_text": instruction["text"],
                "instruction_tokens": _tokenize(instruction["text"], vocabulary),
            }
            episode.setdefault("info", {}).update(
                {
                    "agent_vln_route_id": job.route_id,
                    "instruction_style": style,
                    "instruction_generator": model,
                }
            )
            expanded[job.split].append(episode)

    for split in splits:
        _write_gzip_json(
            output_root / split / f"{split}.json.gz",
            {
                "instruction_vocab": source_documents[split]["instruction_vocab"],
                "episodes": expanded[split],
            },
        )

    status_counts: dict[str, int] = {}
    for route in routes:
        status = route["visual_validation"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    source_manifest = input_root / "manifest.json"
    manifest = {
        "schema_version": 2,
        "name": "agent_vln_r2r_local_gpt56terra_v2",
        "source": {
            "path": str(input_root),
            "manifest_sha256": _file_sha256(source_manifest),
        },
        "generation": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "styles": list(STYLE_ORDER),
        },
        "curation": dict(curation or {}),
        "routes": routes,
    }
    summary = {
        "schema_version": 2,
        "name": manifest["name"],
        "route_count": len(routes),
        "candidate_route_count": int(
            (curation or {}).get("candidate_routes", len(routes))
        ),
        "episode_count": sum(len(values) for values in expanded.values()),
        "instruction_count": len(routes) * len(STYLE_ORDER),
        "splits": {
            split: {
                "routes": sum(job.split == split for job in jobs),
                "episodes": len(expanded[split]),
                "styles": {
                    style: sum(
                        episode["info"]["instruction_style"] == style
                        for episode in expanded[split]
                    )
                    for style in STYLE_ORDER
                },
            }
            for split in splits
        },
        "visual_validation": dict(sorted(status_counts.items())),
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "summary.json", summary)
    return summary


def _rewrite(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        args.concurrency < 1
        or args.retries < 0
        or args.retry_backoff < 0
        or args.min_final_distance < 0
    ):
        raise ValueError("concurrency must be positive and retry settings non-negative")
    input_root, output_root = args.input.resolve(), args.output.resolve()
    jobs = load_jobs(input_root, output_root, limit=args.limit)
    generated, failures = asyncio.run(generate_all(jobs, output_root, args))
    if failures:
        _write_json(output_root / "failures.json", {"failures": failures})
        raise RuntimeError(
            f"{len(failures)} route rewrites failed; rerun the command to resume"
        )
    (output_root / "failures.json").unlink(missing_ok=True)
    if args.limit is not None:
        return {"generated": len(generated), "selected": len(jobs), "materialized": False}
    curation = None
    if args.final_routes is not None:
        jobs, curation = select_final_jobs(
            jobs,
            generated,
            count=args.final_routes,
            min_distance=args.min_final_distance,
        )
        _write_json(output_root / "curation.json", curation)
    return materialize(
        jobs,
        generated,
        input_root=input_root,
        output_root=output_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        curation=curation,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the local AgentVLN dataset")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("build", help="select short R2R routes")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--scenes-root", type=Path, default=Path("data/scene_datasets"))
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--count", type=int, default=100)
    command.add_argument("--seed", type=int, default=20260904)
    command.add_argument("--max-distance", type=float, default=5.5)
    command.add_argument("--max-vertical-span", type=float, default=0.35)
    command.add_argument("--max-viewpoints", type=int, default=6)
    command.add_argument("--max-large-turns", type=int, default=2)
    command.add_argument("--max-turn", type=float, default=135.0)
    command.add_argument("--max-per-scene", type=int, default=4)
    command.add_argument("--max-instruction-words", type=int, default=40)
    command.add_argument("--validate-navmesh", action="store_true")
    command.set_defaults(handler=build)

    command = commands.add_parser("reserve", help="add balanced replacement routes")
    command.add_argument("--base", type=Path, required=True)
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--scenes-root", type=Path, default=Path("data/scene_datasets"))
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--per-pattern", type=int, default=5)
    command.add_argument("--max-distance", type=float, default=6.0)
    command.add_argument("--max-per-scene", type=int, default=5)
    command.add_argument("--seed", type=int, default=20260904)
    command.set_defaults(handler=extend)

    command = commands.add_parser("render", help="sample ordered RGB route views")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--scenes-root", type=Path, default=Path("data/scene_datasets"))
    command.add_argument("--gpu-device-id", type=int, default=0)
    command.add_argument("--width", type=int, default=640)
    command.add_argument("--height", type=int, default=480)
    command.add_argument("--hfov", type=float, default=79.0)
    command.add_argument("--sensor-height", type=float, default=1.25)
    command.add_argument("--max-frames", type=int, default=6)
    command.add_argument("--limit", type=int)
    command.add_argument("--num-shards", type=int, default=1)
    command.add_argument("--shard-index", type=int, default=0)
    command.add_argument("--overwrite", action="store_true")
    command.add_argument(
        "--show-native-logs", dest="quiet_native_logs", action="store_false"
    )
    command.set_defaults(handler=render_dataset, quiet_native_logs=True)

    command = commands.add_parser("rewrite", help="generate three grounded instructions")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--model", default="gpt-5.6-terra")
    command.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    command.add_argument("--image-detail", choices=("low", "high", "auto"), default="high")
    command.add_argument("--concurrency", type=int, default=4)
    command.add_argument("--api-timeout", type=float, default=300.0)
    command.add_argument("--retries", type=int, default=2)
    command.add_argument("--retry-backoff", type=float, default=2.0)
    command.add_argument("--limit", type=int)
    command.add_argument("--final-routes", type=int)
    command.add_argument("--min-final-distance", type=float, default=0.0)
    command.set_defaults(handler=_rewrite)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "render" and (
        args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards
    ):
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.command == "reserve" and (
        args.per_pattern < 1 or args.max_per_scene < 1
    ):
        raise ValueError("reserve and per-scene counts must be positive")
    print(json.dumps(args.handler(args), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import pytest

from utils.datasets.agent_vln.pipeline import (
    Candidate,
    InstructionVariant,
    RewriteJob,
    _initial_alignment,
    _route_geometry,
    _tokenize,
    collect_candidates,
    instruction_style,
    safe_route_id,
    select_final_jobs,
    select_reserves,
    split_sizes,
    uniform_indices,
    validate_generation,
    validate_route_generation,
)


def _episode(episode_id: int, text: str) -> dict:
    return {
        "episode_id": episode_id,
        "trajectory_id": 7,
        "scene_id": "mp3d/scene/scene.glb",
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [3.0, 0.0, 1.0], "radius": 3.0}],
        "info": {"geodesic_distance": 4.0},
        "reference_path": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
        ],
        "instruction": {"instruction_text": text, "instruction_tokens": [1, 2]},
    }


def _candidate(trajectory_id: str, scene_id: str, pattern: str) -> Candidate:
    instruction = InstructionVariant(
        f"source:{trajectory_id}",
        "Walk to the nearby doorway.",
        (1, 2, 3),
        5,
        "concise",
    )
    return Candidate(
        trajectory_id,
        {"scene_id": scene_id},
        (instruction,),
        pattern,
        (),
        {},
    )


def _generation() -> dict:
    return {
        "route_check": {
            "status": "grounded",
            "notes": "The ordered views agree with the route.",
            "verified_landmarks": ["doorway", "table"],
        },
        "instructions": [
            {
                "style": "concise",
                "text": "Go through the doorway and stop beside the table.",
            },
            {
                "style": "natural",
                "text": (
                    "Walk through the nearby doorway, then continue into the room "
                    "and stop beside the table."
                ),
            },
            {
                "style": "landmark_rich",
                "text": (
                    "Head through the open doorway into the next room, continue "
                    "straight past the chairs, and finish beside the table near "
                    "the far wall."
                ),
            },
        ],
    }


def test_route_selection_primitives() -> None:
    document = {
        "episodes": [
            _episode(1, "Walk through the door and stop by the table."),
            _episode(2, "Go straight, turn toward the doorway, and wait by the table."),
            _episode(3, "Enter the nearby room and stand next to the table on your right."),
        ]
    }
    candidates, rejected = collect_candidates(document)
    assert not rejected and len(candidates) == 1
    assert candidates[0].pattern == "one_turn"
    assert {"doorway", "room", "landmark_relative", "right"} <= set(
        candidates[0].semantic_tags
    )
    assert _route_geometry(_episode(1, "Walk forward."))["initial_turn_degrees"] == 90
    assert _initial_alignment(_episode(1, "Walk forward.")) == {
        "direction": "right",
        "degrees": 90.0,
    }
    assert [instruction_style(n) for n in (14, 15, 26)] == [
        "concise",
        "standard",
        "detailed",
    ]
    assert split_sizes(100) == {"debug": 60, "dev": 20, "test": 20}


def test_reserves_preserve_scene_split_and_pattern_balance() -> None:
    splits = {"debug": "debug.glb", "dev": "dev.glb", "test": "test.glb"}
    base = [
        {
            "source": {"trajectory_id": f"base-{split}"},
            "scene_id": scene,
            "split": split,
        }
        for split, scene in splits.items()
    ]
    candidates = [
        _candidate(f"base-{split}", scene, "low_turn")
        for split, scene in splits.items()
    ] + [
        _candidate(f"reserve-{split}-{pattern}", scene, pattern)
        for split, scene in splits.items()
        for pattern in ("low_turn", "one_turn", "two_turn")
    ]
    selected = select_reserves(candidates, base, per_pattern=1, max_per_scene=4, seed=7)
    for split, scene in splits.items():
        assert {item.pattern for item in selected[split]} == {
            "low_turn",
            "one_turn",
            "two_turn",
        }
        assert {item.scene_id for item in selected[split]} == {scene}


def test_route_frame_names_and_sampling() -> None:
    assert uniform_indices(4, 6) == [0, 1, 2, 3]
    assert uniform_indices(9, 4) == [0, 3, 5, 8]
    assert safe_route_id("agent_vln:debug:0001") == "agent_vln_debug_0001"


def test_generation_validation_and_tokenization() -> None:
    result = validate_generation(_generation())
    assert [item["style"] for item in result["instructions"]] == [
        "concise",
        "natural",
        "landmark_rich",
    ]
    value = _generation()
    value["instructions"][0]["text"] = (
        "Follow the route shown in the first image and stop there."
    )
    with pytest.raises(ValueError, match="dataset media"):
        validate_generation(value)
    value = _generation()
    value["instructions"][0]["text"] = (
        "Pass the framed picture, turn left, and stop beside the table."
    )
    assert "framed picture" in validate_generation(value)["instructions"][0]["text"]
    vocabulary = {"UNK_INDEX": 1, "word2idx_dict": {"go": 10, "left": 11, ".": 12}}
    assert _tokenize("Go left.", vocabulary) == [10, 11, 12]

    value = _generation()
    with pytest.raises(ValueError, match="initial right turn"):
        validate_route_generation(value, _episode(1, "Walk forward."))
    for item in value["instructions"]:
        item["text"] = "Turn right, " + item["text"][0].lower() + item["text"][1:]
    assert validate_route_generation(value, _episode(1, "Walk forward."))


def test_final_selection_replaces_conflicts_with_same_pattern() -> None:
    jobs = [
        RewriteJob(
            {
                "episode_id": f"agent_vln:{split}:{pattern}:{index}",
                "split": split,
                "route_pattern": pattern,
                "geometry": {"source_geodesic_distance_m": 4.0 + index / 4},
            },
            {},
            {},
            str(index),
        )
        for split in ("debug", "dev", "test")
        for pattern in ("low_turn", "one_turn", "two_turn")
        for index in range(4)
    ]
    generations = {
        job.route_id: {
            "result": {
                "route_check": {
                    "status": "conflict" if job.route_id.endswith(":0") else "grounded"
                }
            }
        }
        for job in jobs
    }
    selected, curation = select_final_jobs(
        jobs, generations, count=9, min_distance=4.5
    )
    assert len(selected) == 9
    assert all(job.route_id.rsplit(":", 1)[1] in {"2", "3"} for job in selected)
    assert len(curation["excluded_conflicts"]) == 9
    assert curation["minimum_geodesic_distance_m"] == 4.5

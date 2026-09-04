from __future__ import annotations

from utils.datasets.agent_vln.build import (
    _route_geometry,
    collect_candidates,
    instruction_style,
    split_sizes,
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


def test_collect_candidates_groups_human_variants_by_trajectory() -> None:
    document = {
        "episodes": [
            _episode(1, "Walk through the door and stop by the table."),
            _episode(2, "Go straight, turn toward the doorway, and wait beside the table."),
            _episode(3, "Enter the nearby room and stand next to the table on your right."),
        ]
    }

    candidates, rejected = collect_candidates(document)

    assert not rejected
    assert len(candidates) == 1
    assert len(candidates[0].variants) == 3
    assert candidates[0].pattern == "one_turn"
    assert {"doorway", "room", "landmark_relative", "right"} <= set(
        candidates[0].semantic_tags
    )


def test_style_and_default_split_are_explicit() -> None:
    assert instruction_style(14) == "concise"
    assert instruction_style(15) == "standard"
    assert instruction_style(26) == "detailed"
    assert split_sizes(100) == {"debug": 60, "dev": 20, "test": 20}


def test_geometry_includes_turn_from_start_heading() -> None:
    episode = _episode(1, "Walk forward and stop.")
    episode["start_rotation"] = [0.0, 0.0, 0.0, 1.0]

    geometry = _route_geometry(episode)

    assert geometry["initial_turn_degrees"] == 90.0

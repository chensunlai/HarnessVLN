from __future__ import annotations

import pytest

from utils.datasets.agent_vln.rewrite_instructions import (
    RewriteJob,
    _tokenize,
    select_final_jobs,
    validate_generation,
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


def test_generation_requires_three_bounded_styles() -> None:
    value = validate_generation(_generation())

    assert [item["style"] for item in value["instructions"]] == [
        "concise",
        "natural",
        "landmark_rich",
    ]
    assert all(item["word_count"] > 0 for item in value["instructions"])


def test_generation_rejects_media_references() -> None:
    value = _generation()
    value["instructions"][0]["text"] = (
        "Follow the route shown in the first image and stop there."
    )

    with pytest.raises(ValueError, match="dataset media"):
        validate_generation(value)


def test_generation_allows_picture_as_a_navigation_landmark() -> None:
    value = _generation()
    value["instructions"][0]["text"] = (
        "Pass the framed picture, turn left, and stop beside the table."
    )

    result = validate_generation(value)

    assert "framed picture" in result["instructions"][0]["text"]


def test_generated_text_uses_source_vocabulary_tokens() -> None:
    vocabulary = {
        "UNK_INDEX": 1,
        "word2idx_dict": {"go": 10, "left": 11, ".": 12},
    }

    assert _tokenize("Go left.", vocabulary) == [10, 11, 12]


def test_final_selection_replaces_conflicts_with_same_pattern() -> None:
    jobs = [
        RewriteJob(
            {
                "episode_id": f"agent_vln:{split}:{pattern}:{index}",
                "split": split,
                "route_pattern": pattern,
            },
            {},
            {},
            str(index),
        )
        for split in ("debug", "dev", "test")
        for pattern in ("low_turn", "one_turn", "two_turn")
        for index in range(3)
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

    selected, curation = select_final_jobs(jobs, generations, count=9)

    assert len(selected) == 9
    assert all(not job.route_id.endswith(":0") for job in selected)
    assert len(curation["excluded_conflicts"]) == 9

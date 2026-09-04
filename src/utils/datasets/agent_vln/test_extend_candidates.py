from __future__ import annotations

from utils.datasets.agent_vln.build import Candidate, InstructionVariant
from utils.datasets.agent_vln.extend_candidates import select_reserves


def _candidate(trajectory_id: str, scene_id: str, pattern: str) -> Candidate:
    instruction = InstructionVariant(
        source_episode_id=f"source:{trajectory_id}",
        text="Walk to the nearby doorway.",
        tokens=(1, 2, 3),
        word_count=5,
        style="concise",
    )
    return Candidate(
        trajectory_id=trajectory_id,
        representative={"scene_id": scene_id},
        variants=(instruction,),
        pattern=pattern,
        semantic_tags=(),
        geometry={},
    )


def test_reserves_preserve_scene_split_and_balance_patterns() -> None:
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

    selected = select_reserves(
        candidates,
        base,
        per_pattern=1,
        max_per_scene=4,
        seed=7,
    )

    for split, scene in splits.items():
        assert {candidate.pattern for candidate in selected[split]} == {
            "low_turn",
            "one_turn",
            "two_turn",
        }
        assert {candidate.scene_id for candidate in selected[split]} == {scene}
        assert all(
            candidate.trajectory_id != f"base-{split}"
            for candidate in selected[split]
        )

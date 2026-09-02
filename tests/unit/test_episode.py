from __future__ import annotations

import pytest

from domain import NavigationEpisode
from domain.errors import HarnessError


def test_instruction_is_an_open_mapping() -> None:
    source = {
        "type": "target_img",
        "image": "goal.png",
        "bench_extension": {"value": 3},
    }
    episode = NavigationEpisode("image-goal", source)
    source["image"] = "changed.png"

    assert episode.instruction == {
        "type": "target_img",
        "image": "goal.png",
        "bench_extension": {"value": 3},
    }
    assert episode.as_dict()["instruction"] == episode.instruction
    assert NavigationEpisode("empty", {}).instruction == {}


def test_instruction_rejects_non_mapping_values() -> None:
    with pytest.raises(HarnessError, match="instruction must be an object"):
        NavigationEpisode("text", "walk forward")  # type: ignore[arg-type]

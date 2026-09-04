from __future__ import annotations

from utils.datasets.agent_vln.render_paths import safe_route_id, uniform_indices


def test_uniform_indices_keep_endpoints() -> None:
    assert uniform_indices(4, 6) == [0, 1, 2, 3]
    assert uniform_indices(9, 4) == [0, 3, 5, 8]


def test_route_id_is_safe_for_a_directory() -> None:
    assert safe_route_id("agent_vln:debug:0001") == "agent_vln_debug_0001"

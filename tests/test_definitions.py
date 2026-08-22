from __future__ import annotations

import pytest

from grokking_lab.definitions import (
    Graph,
    compute_definitions,
    graph_from_record,
    path_excess,
    safe_paths,
)


def branching_graph() -> Graph:
    return Graph(
        num_nodes=7,
        edges=((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)),
        values=(0.6, 0.4, 0.6, 0.4, 0.7, 0.3, 0.7, 0.3),
    )


def test_definition_01_on_known_junction() -> None:
    values = compute_definitions(branching_graph())
    assert values["definition_01"] == pytest.approx(0.4)
    assert 0 <= values["definition_03"] <= 1
    assert values["definition_04"] >= 1


def test_definition_05_is_definition_04_without_length_weighting() -> None:
    values = compute_definitions(branching_graph())
    assert values["definition_04"] == pytest.approx(5.8)
    assert values["definition_05"] == pytest.approx(2.6)
    assert 1 <= values["definition_05"] < values["definition_04"]


def test_safe_path_excess() -> None:
    graph = branching_graph()
    # Edges 2 and 4 carry at least 0.3 together: 0.6 + 0.7 - 1.0.
    assert path_excess(graph, (2, 4)) == pytest.approx(0.3)
    assert (2, 4) in safe_paths(graph)
    assert (3, 5) not in safe_paths(graph)


def test_rejects_nonconserved_graph() -> None:
    with pytest.raises(ValueError, match="conservation"):
        Graph(3, ((0, 1), (1, 2)), (1.0, 0.5))


def test_fixed_point_record_decodes_to_unit_flow() -> None:
    graph = graph_from_record(
        {
            "flow_scale": 10_000_000,
            "edges": [[0, 1, 6_000_000], [0, 2, 4_000_000], [1, 3, 6_000_000], [2, 3, 4_000_000]],
        },
        4,
    )
    assert graph.value == pytest.approx(1.0)
    assert graph.values == pytest.approx((0.6, 0.4, 0.6, 0.4))

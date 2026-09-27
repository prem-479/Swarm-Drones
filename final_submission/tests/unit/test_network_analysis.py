from __future__ import annotations

import numpy as np

from uavx.communication import GCS_NODE_ID
from uavx.core.models import NetworkGraph
from uavx.network import NetworkGraphAnalyzer


def make_graph(edges: set[tuple[int, int]]) -> NetworkGraph:
    nodes = sorted(
        {
            node_id
            for edge in edges
            for node_id in edge
        }
    )

    graph = NetworkGraph(
        timestamp_s=0.0,
        nodes=nodes,
        edges=edges,
        links={},
        laplacian=np.zeros((len(nodes), len(nodes))),
    )
    graph.update_laplacian()
    return graph


def test_connected_components() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (0, 1),
            (1, 2),
            (3, 4),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    components = analyzer.connected_components()

    assert any(component == {-1, 0, 1, 2} for component in components)
    assert any(component == {3, 4} for component in components)


def test_shortest_path() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (0, 1),
            (1, 2),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    assert analyzer.shortest_path(GCS_NODE_ID, 2) == [-1, 0, 1, 2]


def test_articulation_points() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (0, 1),
            (1, 2),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    assert analyzer.articulation_points() == {0, 1}


def test_bridges() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (0, 1),
            (1, 2),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    assert analyzer.bridges() == {
        (-1, 0),
        (0, 1),
        (1, 2),
    }


def test_degree() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 2),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    assert analyzer.node_degree(-1) == 2
    assert analyzer.node_degree(1) == 3


def test_path_redundancy() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (0, 1),
            (1, 2),
            (-1, 2),
            (0, 2),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    assert analyzer.path_redundancy(GCS_NODE_ID, 2) == 2


def test_gcs_reachability() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (0, 1),
            (3, 4),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    assert analyzer.gcs_reachability() == {-1, 0, 1} - {-1}
    assert analyzer.gcs_reachability({0, 1, 3}) == {0, 1}


def test_partition_detection() -> None:
    graph = make_graph(
        {
            (-1, 0),
            (0, 1),
            (3, 4),
        }
    )

    analyzer = NetworkGraphAnalyzer(graph)

    assert analyzer.is_partitioned({0, 1}) is False
    assert analyzer.is_partitioned({0, 1, 3}) is True

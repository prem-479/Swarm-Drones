from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

from uavx.communication import GCS_NODE_ID
from uavx.core.models import NetworkGraph


@dataclass(frozen=True)
class NetworkAnalysis:
    connected_components: list[set[int]]
    articulation_points: set[int]
    bridges: set[tuple[int, int]]
    degrees: dict[int, int]
    gcs_reachable: set[int]
    partitioned: bool


class NetworkGraphAnalyzer:
    """Analyze the active communication topology."""

    def __init__(self, graph: NetworkGraph) -> None:
        self.graph = graph
        self.nx_graph = self._build_nx_graph()

    def _build_nx_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.graph.nodes)

        for source_id, target_id in self.graph.edges:
            graph.add_edge(source_id, target_id)

        return graph

    def connected_components(self) -> list[set[int]]:
        return [set(component) for component in nx.connected_components(self.nx_graph)]

    def shortest_path(self, source_id: int, target_id: int) -> list[int]:
        return nx.shortest_path(
            self.nx_graph,
            source=source_id,
            target=target_id,
        )

    def articulation_points(self) -> set[int]:
        return {
            node_id
            for node_id in nx.articulation_points(self.nx_graph)
            if node_id != GCS_NODE_ID
        }

    def bridges(self) -> set[tuple[int, int]]:
        return {
            tuple(sorted(edge))
            for edge in nx.bridges(self.nx_graph)
        }

    def node_degree(self, node_id: int) -> int:
        return int(self.nx_graph.degree(node_id))

    def node_degrees(self) -> dict[int, int]:
        return {
            node_id: int(degree)
            for node_id, degree in self.nx_graph.degree()
        }

    def path_redundancy(self, source_id: int, target_id: int) -> int:
        """Return the number of edge-disjoint paths between two nodes."""
        if source_id not in self.nx_graph or target_id not in self.nx_graph:
            return 0

        if source_id == target_id:
            return 0

        if not nx.has_path(self.nx_graph, source_id, target_id):
            return 0

        return int(
            nx.edge_connectivity(
                self.nx_graph,
                source_id,
                target_id,
            )
        )

    def gcs_reachability(
        self,
        node_ids: Iterable[int] | None = None,
    ) -> set[int]:
        if GCS_NODE_ID not in self.nx_graph:
            return set()

        reachable = nx.node_connected_component(
            self.nx_graph,
            GCS_NODE_ID,
        )

        if node_ids is None:
            return {
                node_id
                for node_id in reachable
                if node_id != GCS_NODE_ID
            }

        requested = set(node_ids)
        return requested.intersection(reachable)

    def is_partitioned(
        self,
        required_node_ids: Iterable[int] | None = None,
    ) -> bool:
        if GCS_NODE_ID not in self.nx_graph:
            return True

        if required_node_ids is None:
            required_node_ids = [
                node_id
                for node_id in self.nx_graph.nodes
                if node_id != GCS_NODE_ID
            ]

        required = set(required_node_ids)

        if not required:
            return False

        reachable = self.gcs_reachability(required)

        return reachable != required

    def analyze(
        self,
        required_node_ids: Iterable[int] | None = None,
    ) -> NetworkAnalysis:
        return NetworkAnalysis(
            connected_components=self.connected_components(),
            articulation_points=self.articulation_points(),
            bridges=self.bridges(),
            degrees=self.node_degrees(),
            gcs_reachable=self.gcs_reachability(required_node_ids),
            partitioned=self.is_partitioned(required_node_ids),
        )

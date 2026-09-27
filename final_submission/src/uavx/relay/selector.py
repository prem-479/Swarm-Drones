from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from uavx.communication import GCS_NODE_ID
from uavx.core.models import NetworkGraph, UAVRole, UAVState
from uavx.network import NetworkAnalysis


@dataclass(frozen=True)
class RelayScoreWeights:
    """Configurable weights for the baseline relay heuristic."""

    connectivity_contribution: float = 2.0
    backbone_proximity: float = 1.0
    link_margin: float = 1.5
    centrality: float = 1.0
    articulation_risk: float = 1.0
    battery_state: float = 1.0
    task_burden: float = 1.0
    future_mobility: float = 0.5
    recovery_value: float = 1.0


@dataclass(frozen=True)
class RelayCandidateScore:
    uav_id: int
    total_score: float
    connectivity_contribution: float
    backbone_proximity: float
    link_margin: float
    centrality: float
    articulation_risk: float
    battery_state: float
    task_burden: float
    future_mobility: float
    recovery_value: float
    bridgeable_terminal_ids: tuple[int, ...]


class RelaySelector:
    """
    Baseline dynamic relay-selection heuristic.

    A candidate is valuable when it can bridge terminal UAVs to the
    communication backbone while preserving link margin, battery,
    mobility headroom and recovery flexibility.

    This is intentionally heuristic rather than an exact Steiner solver.
    """

    def __init__(
        self,
        max_range_m: float,
        max_speed_mps: float,
        weights: RelayScoreWeights | None = None,
    ) -> None:
        if max_range_m <= 0:
            raise ValueError("max_range_m must be > 0")

        if max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be > 0")

        self.max_range_m = float(max_range_m)
        self.max_speed_mps = float(max_speed_mps)
        self.weights = weights or RelayScoreWeights()

    @staticmethod
    def _edge_key(source_id: int, target_id: int) -> tuple[int, int]:
        return tuple(sorted((source_id, target_id)))

    def _active_link(
        self,
        graph: NetworkGraph,
        source_id: int,
        target_id: int,
    ):
        return graph.links.get(self._edge_key(source_id, target_id))

    def _link_margin(
        self,
        graph: NetworkGraph,
        source_id: int,
        target_id: int,
    ) -> float:
        link = self._active_link(graph, source_id, target_id)

        if link is None or not link.availability:
            return 0.0

        margin = 1.0 - (link.distance_m / self.max_range_m)

        return float(np.clip(margin, 0.0, 1.0))

    def _backbone_metrics(
        self,
        candidate_id: int,
        backbone_nodes: list[int],
        graph: NetworkGraph,
    ) -> tuple[float, float]:
        margins = []
        distances = []

        for backbone_id in backbone_nodes:
            link = self._active_link(graph, candidate_id, backbone_id)

            if link is None or not link.availability:
                continue

            distances.append(link.distance_m)
            margins.append(
                float(
                    np.clip(
                        1.0 - link.distance_m / self.max_range_m,
                        0.0,
                        1.0,
                    )
                )
            )

        if not distances:
            return 0.0, 0.0

        min_distance = min(distances)

        proximity = float(
            np.clip(
                1.0 - min_distance / self.max_range_m,
                0.0,
                1.0,
            )
        )

        return proximity, max(margins)

    def _bridgeable_terminals(
        self,
        candidate_id: int,
        backbone_nodes: list[int],
        terminal_nodes: list[int],
        graph: NetworkGraph,
    ) -> tuple[int, ...]:
        backbone_connected = any(
            self._active_link(graph, candidate_id, backbone_id) is not None
            and self._active_link(
                graph,
                candidate_id,
                backbone_id,
            ).availability
            for backbone_id in backbone_nodes
        )

        if not backbone_connected:
            return ()

        bridgeable = []

        for terminal_id in terminal_nodes:
            if terminal_id == candidate_id:
                continue

            terminal_link = self._active_link(
                graph,
                candidate_id,
                terminal_id,
            )

            if terminal_link is not None and terminal_link.availability:
                bridgeable.append(terminal_id)

        return tuple(sorted(set(bridgeable)))

    def score_candidate(
        self,
        candidate: UAVState,
        uavs: dict[int, UAVState],
        graph: NetworkGraph,
        analysis: NetworkAnalysis,
        backbone_nodes: Iterable[int],
        terminal_nodes: Iterable[int],
    ) -> RelayCandidateScore:
        backbone = list(dict.fromkeys(backbone_nodes))
        terminals = list(dict.fromkeys(terminal_nodes))

        bridgeable = self._bridgeable_terminals(
            candidate.uav_id,
            backbone,
            terminals,
            graph,
        )

        terminal_count = max(len(terminals), 1)

        connectivity_contribution = len(bridgeable) / terminal_count

        backbone_proximity, backbone_margin = self._backbone_metrics(
            candidate.uav_id,
            backbone,
            graph,
        )

        terminal_margins = [
            self._link_margin(
                graph,
                candidate.uav_id,
                terminal_id,
            )
            for terminal_id in bridgeable
        ]

        all_margins = [backbone_margin, *terminal_margins]
        link_margin = (
            float(np.mean([value for value in all_margins if value > 0.0]))
            if any(value > 0.0 for value in all_margins)
            else 0.0
        )

        node_count = max(len(analysis.degrees) - 1, 1)
        centrality = float(
            np.clip(
                analysis.degrees.get(candidate.uav_id, 0) / node_count,
                0.0,
                1.0,
            )
        )

        articulation_risk = (
            1.0 if candidate.uav_id in analysis.articulation_points else 0.0
        )

        battery_state = float(np.clip(candidate.battery.soc, 0.0, 1.0))

        task_burden = 1.0 if candidate.assigned_task_id is not None else 0.0

        current_speed = float(np.linalg.norm(candidate.velocity))

        future_mobility = float(
            np.clip(
                1.0 - current_speed / self.max_speed_mps,
                0.0,
                1.0,
            )
        )

        neighbor_count = len(candidate.neighbors)
        recovery_value = float(
            np.clip(
                max(neighbor_count - 1, 0) / 4.0,
                0.0,
                1.0,
            )
        )

        w = self.weights

        total_score = (
            w.connectivity_contribution * connectivity_contribution
            + w.backbone_proximity * backbone_proximity
            + w.link_margin * link_margin
            + w.centrality * centrality
            - w.articulation_risk * articulation_risk
            + w.battery_state * battery_state
            - w.task_burden * task_burden
            + w.future_mobility * future_mobility
            + w.recovery_value * recovery_value
        )

        return RelayCandidateScore(
            uav_id=candidate.uav_id,
            total_score=float(total_score),
            connectivity_contribution=float(connectivity_contribution),
            backbone_proximity=float(backbone_proximity),
            link_margin=float(link_margin),
            centrality=float(centrality),
            articulation_risk=float(articulation_risk),
            battery_state=float(battery_state),
            task_burden=float(task_burden),
            future_mobility=float(future_mobility),
            recovery_value=float(recovery_value),
            bridgeable_terminal_ids=bridgeable,
        )

    def rank_candidates(
        self,
        uavs: dict[int, UAVState],
        graph: NetworkGraph,
        analysis: NetworkAnalysis,
        backbone_nodes: Iterable[int] | None = None,
        terminal_nodes: Iterable[int] | None = None,
    ) -> list[RelayCandidateScore]:
        if backbone_nodes is None:
            backbone_nodes = [GCS_NODE_ID] + [
                uav_id
                for uav_id, uav in uavs.items()
                if uav.role == UAVRole.RELAY
            ]

        if terminal_nodes is None:
            terminal_nodes = [
                uav_id
                for uav_id, uav in uavs.items()
                if uav.role == UAVRole.SURVEY
            ]

        backbone = set(backbone_nodes)
        terminals = set(terminal_nodes)

        candidates = []

        for uav in uavs.values():
            if uav.uav_id in backbone:
                continue

            if uav.uav_id in terminals:
                continue

            if uav.failure_state:
                continue

            if uav.role in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }:
                continue

            score = self.score_candidate(
                candidate=uav,
                uavs=uavs,
                graph=graph,
                analysis=analysis,
                backbone_nodes=backbone,
                terminal_nodes=terminals,
            )

            candidates.append(score)

        return sorted(
            candidates,
            key=lambda item: (-item.total_score, item.uav_id),
        )

    def select_relays(
        self,
        uavs: dict[int, UAVState],
        graph: NetworkGraph,
        analysis: NetworkAnalysis,
        max_relays: int = 1,
        backbone_nodes: Iterable[int] | None = None,
        terminal_nodes: Iterable[int] | None = None,
    ) -> list[int]:
        if max_relays <= 0:
            return []

        ranked = self.rank_candidates(
            uavs=uavs,
            graph=graph,
            analysis=analysis,
            backbone_nodes=backbone_nodes,
            terminal_nodes=terminal_nodes,
        )

        selected: list[int] = []

        for candidate in ranked:
            if candidate.connectivity_contribution <= 0.0:
                continue

            selected.append(candidate.uav_id)

            if len(selected) >= max_relays:
                break

        return selected

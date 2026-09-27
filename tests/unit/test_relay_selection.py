from __future__ import annotations

import numpy as np

from uavx.communication import GCS_NODE_ID, CommunicationEngine
from uavx.core import BatteryState
from uavx.core.models import NetworkGraph, UAVRole, UAVState
from uavx.network import NetworkGraphAnalyzer
from uavx.relay import RelayScoreWeights, RelaySelector


def make_uav(
    uav_id: int,
    x: float,
    role: UAVRole,
    battery: float = 1.0,
    task_id: int | None = None,
    speed: float = 0.0,
) -> UAVState:
    return UAVState(
        uav_id=uav_id,
        position=np.array([x, 0.0, 20.0]),
        velocity=np.array([speed, 0.0, 0.0]),
        acceleration=np.zeros(3),
        role=role,
        battery=BatteryState(soc=battery),
        assigned_task_id=task_id,
        home_position=np.array([0.0, 0.0, 20.0]),
    )


def build_graph(uavs: dict[int, UAVState]) -> NetworkGraph:
    communication = CommunicationEngine(
        config=_communication_config(),
        seed=42,
    )

    gcs_position = np.array([0.0, 0.0, 20.0])

    positions = {
        uav_id: uav.position.copy()
        for uav_id, uav in uavs.items()
    }

    return communication.build_graph(
        positions=positions,
        timestamp_s=0.0,
        gcs_position=gcs_position,
    )


def _communication_config():
    from uavx.core.config import CommunicationConfig

    return CommunicationConfig(
        max_range_m=100.0,
        model="binary_range",
        packet_delivery_probability=1.0,
        latency_ms=0.0,
    )


def test_candidate_can_bridge_backbone_to_terminal() -> None:
    uavs = {
        0: make_uav(0, 50.0, UAVRole.RESERVE),
        1: make_uav(1, 150.0, UAVRole.SURVEY),
        2: make_uav(2, 300.0, UAVRole.RESERVE),
    }

    graph = build_graph(uavs)
    analysis = NetworkGraphAnalyzer(graph).analyze()

    selector = RelaySelector(
        max_range_m=100.0,
        max_speed_mps=5.0,
    )

    score = selector.score_candidate(
        candidate=uavs[0],
        uavs=uavs,
        graph=graph,
        analysis=analysis,
        backbone_nodes=[GCS_NODE_ID],
        terminal_nodes=[1],
    )

    assert score.bridgeable_terminal_ids == (1,)
    assert score.connectivity_contribution == 1.0
    assert score.backbone_proximity == 0.5
    assert score.link_margin > 0.0


def test_candidate_without_bridge_has_zero_connectivity_contribution() -> None:
    uavs = {
        0: make_uav(0, 250.0, UAVRole.RESERVE),
        1: make_uav(1, 350.0, UAVRole.SURVEY),
    }

    graph = build_graph(uavs)
    analysis = NetworkGraphAnalyzer(graph).analyze()

    selector = RelaySelector(
        max_range_m=100.0,
        max_speed_mps=5.0,
    )

    score = selector.score_candidate(
        candidate=uavs[0],
        uavs=uavs,
        graph=graph,
        analysis=analysis,
        backbone_nodes=[GCS_NODE_ID],
        terminal_nodes=[1],
    )

    assert score.connectivity_contribution == 0.0
    assert score.bridgeable_terminal_ids == ()


def test_battery_and_task_burden_affect_score() -> None:
    base = make_uav(0, 50.0, UAVRole.RESERVE)
    burdened = make_uav(
        0,
        50.0,
        UAVRole.RESERVE,
        battery=0.4,
        task_id=99,
    )

    uavs_base = {
        0: base,
        1: make_uav(1, 150.0, UAVRole.SURVEY),
    }

    uavs_burdened = {
        0: burdened,
        1: make_uav(1, 150.0, UAVRole.SURVEY),
    }

    graph_base = build_graph(uavs_base)
    graph_burdened = build_graph(uavs_burdened)

    analysis_base = NetworkGraphAnalyzer(graph_base).analyze()
    analysis_burdened = NetworkGraphAnalyzer(graph_burdened).analyze()

    selector = RelaySelector(
        max_range_m=100.0,
        max_speed_mps=5.0,
    )

    score_base = selector.score_candidate(
        base,
        uavs_base,
        graph_base,
        analysis_base,
        [GCS_NODE_ID],
        [1],
    )

    score_burdened = selector.score_candidate(
        burdened,
        uavs_burdened,
        graph_burdened,
        analysis_burdened,
        [GCS_NODE_ID],
        [1],
    )

    assert score_burdened.battery_state == 0.4
    assert score_burdened.task_burden == 1.0
    assert score_burdened.total_score < score_base.total_score


def test_articulation_risk_is_penalized() -> None:
    uavs = {
        0: make_uav(0, 50.0, UAVRole.RESERVE),
        1: make_uav(1, 150.0, UAVRole.RESERVE),
        2: make_uav(2, 250.0, UAVRole.SURVEY),
    }

    graph = build_graph(uavs)
    analysis = NetworkGraphAnalyzer(graph).analyze()

    assert 0 in analysis.articulation_points
    assert 1 in analysis.articulation_points

    selector = RelaySelector(
        max_range_m=100.0,
        max_speed_mps=5.0,
    )

    score = selector.score_candidate(
        uavs[0],
        uavs,
        graph,
        analysis,
        [GCS_NODE_ID],
        [2],
    )

    assert score.articulation_risk == 1.0


def test_rank_candidates_is_deterministic() -> None:
    uavs = {
        0: make_uav(0, 50.0, UAVRole.RESERVE, battery=1.0),
        1: make_uav(1, 55.0, UAVRole.RESERVE, battery=0.8),
        2: make_uav(2, 150.0, UAVRole.SURVEY),
    }

    graph = build_graph(uavs)
    analysis = NetworkGraphAnalyzer(graph).analyze()

    selector = RelaySelector(
        max_range_m=100.0,
        max_speed_mps=5.0,
    )

    ranked = selector.rank_candidates(
        uavs=uavs,
        graph=graph,
        analysis=analysis,
        backbone_nodes=[GCS_NODE_ID],
        terminal_nodes=[2],
    )

    assert [score.uav_id for score in ranked] == [0, 1]
    assert ranked[0].total_score >= ranked[1].total_score


def test_select_relays_returns_only_useful_candidates() -> None:
    uavs = {
        0: make_uav(0, 50.0, UAVRole.RESERVE),
        1: make_uav(1, 150.0, UAVRole.SURVEY),
        2: make_uav(2, 350.0, UAVRole.RESERVE),
    }

    graph = build_graph(uavs)
    analysis = NetworkGraphAnalyzer(graph).analyze()

    selector = RelaySelector(
        max_range_m=100.0,
        max_speed_mps=5.0,
    )

    selected = selector.select_relays(
        uavs=uavs,
        graph=graph,
        analysis=analysis,
        max_relays=2,
        backbone_nodes=[GCS_NODE_ID],
        terminal_nodes=[1],
    )

    assert selected == [0]


def test_custom_weights_are_supported() -> None:
    uavs = {
        0: make_uav(0, 50.0, UAVRole.RESERVE),
        1: make_uav(1, 150.0, UAVRole.SURVEY),
    }

    graph = build_graph(uavs)
    analysis = NetworkGraphAnalyzer(graph).analyze()

    selector = RelaySelector(
        max_range_m=100.0,
        max_speed_mps=5.0,
        weights=RelayScoreWeights(
            connectivity_contribution=10.0,
            backbone_proximity=0.0,
            link_margin=0.0,
            centrality=0.0,
            articulation_risk=0.0,
            battery_state=0.0,
            task_burden=0.0,
            future_mobility=0.0,
            recovery_value=0.0,
        ),
    )

    score = selector.score_candidate(
        uavs[0],
        uavs,
        graph,
        analysis,
        [GCS_NODE_ID],
        [1],
    )

    assert score.total_score == 10.0

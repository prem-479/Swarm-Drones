from __future__ import annotations

import numpy as np

from uavx.communication import GCS_NODE_ID, CommunicationEngine
from uavx.core.config import CommunicationConfig
from uavx.core.models import LinkStatus


def make_config(model: str) -> CommunicationConfig:
    return CommunicationConfig(
        max_range_m=100.0,
        model=model,
        packet_delivery_probability=1.0,
        latency_ms=5.0,
    )


def test_binary_range_model() -> None:
    engine = CommunicationEngine(
        config=make_config("binary_range"),
        seed=42,
    )

    link = engine.compute_link(
        0,
        1,
        np.array([0.0, 0.0, 20.0]),
        np.array([80.0, 0.0, 20.0]),
    )

    assert link.availability is True
    assert link.status == LinkStatus.UP
    assert link.packet_delivery_probability == 1.0


def test_binary_range_rejects_link_beyond_100m() -> None:
    engine = CommunicationEngine(
        config=make_config("binary_range"),
        seed=42,
    )

    link = engine.compute_link(
        0,
        1,
        np.array([0.0, 0.0, 20.0]),
        np.array([100.1, 0.0, 20.0]),
    )

    assert link.availability is False
    assert link.status == LinkStatus.DOWN
    assert link.packet_delivery_probability == 0.0


def test_distance_probabilistic_model_degrades_with_distance() -> None:
    engine = CommunicationEngine(
        config=make_config("distance_probabilistic"),
        seed=42,
    )

    near = engine.compute_link(
        0,
        1,
        np.array([0.0, 0.0, 20.0]),
        np.array([20.0, 0.0, 20.0]),
    )

    far = engine.compute_link(
        0,
        1,
        np.array([0.0, 0.0, 20.0]),
        np.array([80.0, 0.0, 20.0]),
    )

    assert near.packet_delivery_probability > far.packet_delivery_probability
    assert far.availability is True


def test_log_distance_model_is_seed_reproducible() -> None:
    position_a = np.array([0.0, 0.0, 20.0])
    position_b = np.array([60.0, 0.0, 20.0])

    first = CommunicationEngine(
        config=make_config("log_distance"),
        seed=123,
    ).compute_link(0, 1, position_a, position_b)

    second = CommunicationEngine(
        config=make_config("log_distance"),
        seed=123,
    ).compute_link(0, 1, position_a, position_b)

    assert first.packet_delivery_probability == second.packet_delivery_probability
    assert first.link_quality == second.link_quality


def test_log_distance_model_respects_hard_range() -> None:
    engine = CommunicationEngine(
        config=make_config("log_distance"),
        seed=42,
    )

    link = engine.compute_link(
        0,
        1,
        np.array([0.0, 0.0, 20.0]),
        np.array([100.1, 0.0, 20.0]),
    )

    assert link.availability is False
    assert link.status == LinkStatus.DOWN
    assert link.packet_delivery_probability == 0.0


def test_graph_includes_gcs() -> None:
    engine = CommunicationEngine(
        config=make_config("binary_range"),
        seed=42,
    )

    graph = engine.build_graph(
        positions={
            0: np.array([0.0, 0.0, 20.0]),
            1: np.array([50.0, 0.0, 20.0]),
            2: np.array([120.0, 0.0, 20.0]),
        },
        timestamp_s=10.0,
        gcs_position=np.array([0.0, 0.0, 20.0]),
    )

    assert GCS_NODE_ID in graph.nodes
    assert 0 in graph.nodes
    assert 1 in graph.nodes
    assert 2 in graph.nodes

    assert (-1, 0) in graph.edges
    assert (-1, 1) in graph.edges
    assert (-1, 2) not in graph.edges

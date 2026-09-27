from __future__ import annotations

import numpy as np

from uavx.communication import GCS_NODE_ID
from uavx.core import load_config
from uavx.simulation import SimulationEngine


CONFIG_PATH = "configs/official_stage1.yaml"


def test_simulation_initializes_live_network() -> None:
    config = load_config(CONFIG_PATH)
    engine = SimulationEngine.from_config(config)

    graph = engine.state.network

    assert GCS_NODE_ID in graph.nodes
    assert len(graph.nodes) == config.uav.count + 1

    # UAV 0 and the GCS start at the same position.
    assert (GCS_NODE_ID, 0) in graph.edges

    # UAV 0 and UAV 1 start 25 m apart.
    assert 1 in engine.state.uavs[0].neighbors
    assert engine.state.uavs[0].communication_quality > 0.0


def test_network_rebuilds_when_uav_moves_out_of_range() -> None:
    config = load_config(CONFIG_PATH)
    engine = SimulationEngine.from_config(config)

    # UAV 1 initially sits 25 m from UAV 0.
    assert 1 in engine.state.uavs[0].neighbors

    # Move UAV 1 far outside the communication formation.
    engine.state.uavs[1].position = np.array(
        [500.0, 500.0, 20.0],
        dtype=float,
    )

    engine.step(1.0)

    graph = engine.state.network

    assert graph.timestamp_s == 1.0
    assert (GCS_NODE_ID, 1) not in graph.edges
    assert 1 not in engine.state.uavs[0].neighbors

    # No active UAV/GCS link exists within 100 m.
    assert engine.state.uavs[1].communication_quality == 0.0

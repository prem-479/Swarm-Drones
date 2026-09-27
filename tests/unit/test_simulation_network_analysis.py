from __future__ import annotations

import numpy as np

from uavx.core import load_config
from uavx.simulation import SimulationEngine


CONFIG_PATH = "configs/official_stage1.yaml"


def test_simulation_initializes_automatic_network_analysis() -> None:
    config = load_config(CONFIG_PATH)
    engine = SimulationEngine.from_config(config)

    analysis = engine.network_analysis

    assert analysis is not None
    assert analysis.partitioned is False

    assert analysis.gcs_reachable == set(range(config.uav.count))

    assert len(analysis.connected_components) == 1

    for uav_id in range(config.uav.count):
        assert analysis.degrees[uav_id] >= 1


def test_network_analysis_updates_after_partition() -> None:
    config = load_config(CONFIG_PATH)
    engine = SimulationEngine.from_config(config)

    assert engine.network_analysis is not None
    assert engine.network_analysis.partitioned is False

    engine.state.uavs[1].position = np.array(
        [500.0, 500.0, 20.0],
        dtype=float,
    )

    engine.step(1.0)

    analysis = engine.network_analysis

    assert analysis is not None
    assert analysis.partitioned is True
    assert 1 not in analysis.gcs_reachable

    assert len(analysis.connected_components) >= 2
    assert not any(
        1 in component and -1 in component
        for component in analysis.connected_components
    )

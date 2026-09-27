import numpy as np

from uavx.core import (
    BatteryState,
    NetworkGraph,
    UAVRole,
    UAVState,
    load_config,
)


CONFIG = "configs/official_stage1.yaml"


def test_config_loads():
    cfg = load_config(CONFIG)

    assert cfg.environment.width_m == 1000.0
    assert cfg.environment.height_m == 1000.0
    assert cfg.uav.max_speed_mps == 5.0
    assert cfg.uav.endurance_s == 1200.0
    assert cfg.communication.max_range_m == 100.0
    assert cfg.simulation.mission_duration_s == 2700.0
    assert cfg.scenario.poi_count == 10


def test_uav_state():
    uav = UAVState(
        uav_id=1,
        position=np.array([0.0, 0.0, 20.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        role=UAVRole.SURVEY,
        battery=BatteryState(),
    )

    assert uav.uav_id == 1
    assert uav.position.shape == (3,)
    assert uav.role == UAVRole.SURVEY


def test_network_laplacian():
    graph = NetworkGraph(
        timestamp_s=0.0,
        nodes=[0, 1, 2],
        edges={(0, 1), (1, 2)},
    )

    graph.update_laplacian()

    expected = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )

    assert np.array_equal(graph.laplacian, expected)
    assert graph.degree(1) == 2

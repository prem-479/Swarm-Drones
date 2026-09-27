from types import SimpleNamespace

import numpy as np

from uavx.core.models import (
    BatteryState,
    UAVRole,
    UAVState,
)
from uavx.simulation.professional_motion import (
    ProfessionalMotionSystem,
)


def make_uav(
    uid: int,
    position,
    role=UAVRole.SURVEY,
):

    return UAVState(
        uav_id=uid,
        position=np.asarray(
            position,
            dtype=float,
        ),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        role=role,
        battery=BatteryState(),
        home_position=np.asarray(
            position,
            dtype=float,
        ),
    )


def test_speed_limit_is_respected():

    system = ProfessionalMotionSystem(
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        min_separation_m=20.0,
        width_m=1000.0,
        height_m=1000.0,
        max_altitude_m=100.0,
    )

    uav = make_uav(
        0,
        [100.0, 100.0, 20.0],
    )

    uav.target = np.array(
        [900.0, 900.0, 20.0]
    )

    state = SimpleNamespace(
        timestamp_s=0.0,
        uavs={0: uav},
    )

    for _ in range(100):

        system.advance(
            state,
            0.1,
        )

        assert np.linalg.norm(
            uav.velocity
        ) <= 5.000001


def test_two_uavs_do_not_cross_hard_boundary():

    system = ProfessionalMotionSystem(
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        min_separation_m=20.0,
        width_m=1000.0,
        height_m=1000.0,
        max_altitude_m=100.0,
    )

    a = make_uav(
        0,
        [200.0, 500.0, 20.0],
    )

    b = make_uav(
        1,
        [260.0, 500.0, 20.0],
    )

    a.target = np.array(
        [900.0, 500.0, 20.0]
    )

    b.target = np.array(
        [0.0, 500.0, 20.0]
    )

    state = SimpleNamespace(
        timestamp_s=0.0,
        uavs={
            0: a,
            1: b,
        },
    )

    minimum = 1e9

    for _ in range(300):

        system.advance(
            state,
            0.1,
        )

        distance = np.linalg.norm(
            a.position - b.position
        )

        minimum = min(
            minimum,
            distance,
        )

    assert minimum >= 20.0


def test_geofence_is_hard():

    system = ProfessionalMotionSystem(
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        min_separation_m=20.0,
        width_m=1000.0,
        height_m=1000.0,
        max_altitude_m=100.0,
    )

    uav = make_uav(
        0,
        [998.0, 998.0, 99.0],
    )

    uav.target = np.array(
        [1500.0, 1500.0, 150.0]
    )

    state = SimpleNamespace(
        timestamp_s=0.0,
        uavs={0: uav},
    )

    for _ in range(100):

        system.advance(
            state,
            0.1,
        )

        assert 0.0 <= uav.position[0] <= 1000.0
        assert 0.0 <= uav.position[1] <= 1000.0
        assert 0.0 <= uav.position[2] <= 100.0

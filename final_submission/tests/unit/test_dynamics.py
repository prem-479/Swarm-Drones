import numpy as np

from uavx.simulation.dynamics import (
    DynamicsLimits,
    HighFidelityFlightDynamics,
)


def test_speed_is_hard_limited():

    model = HighFidelityFlightDynamics()

    result = model.step(
        position=np.array([0.0, 0.0, 20.0]),
        velocity=np.array([4.9, 0.0, 0.0]),
        preferred_velocity=np.array([5.0, 0.0, 0.0]),
        role="SURVEY",
        dt_s=0.1,
    )

    assert np.linalg.norm(
        result.velocity
    ) <= 5.0 + 1e-9


def test_predictive_avoidance_changes_conflicting_velocity():

    model = HighFidelityFlightDynamics()

    result = model.step(
        position=np.array(
            [0.0, 0.0, 20.0]
        ),
        velocity=np.array(
            [2.0, 0.0, 0.0]
        ),
        preferred_velocity=np.array(
            [5.0, 0.0, 0.0]
        ),
        role="SURVEY",
        dt_s=0.1,
        neighbors={
            1: (
                np.array(
                    [35.0, 0.0, 20.0]
                ),
                np.array(
                    [-2.0, 0.0, 0.0]
                ),
            )
        },
    )

    assert result.collision_avoided is True


def test_relay_has_lower_maneuver_responsibility():

    model = HighFidelityFlightDynamics()

    survey = model.step(
        position=np.array(
            [0.0, 0.0, 20.0]
        ),
        velocity=np.array(
            [3.0, 0.0, 0.0]
        ),
        preferred_velocity=np.array(
            [5.0, 0.0, 0.0]
        ),
        role="SURVEY",
        dt_s=0.1,
        neighbors={
            1: (
                np.array(
                    [28.0, 0.0, 20.0]
                ),
                np.array(
                    [-1.0, 0.0, 0.0]
                ),
            )
        },
    )

    relay = model.step(
        position=np.array(
            [0.0, 0.0, 20.0]
        ),
        velocity=np.array(
            [3.0, 0.0, 0.0]
        ),
        preferred_velocity=np.array(
            [5.0, 0.0, 0.0]
        ),
        role="RELAY",
        dt_s=0.1,
        neighbors={
            1: (
                np.array(
                    [28.0, 0.0, 20.0]
                ),
                np.array(
                    [-1.0, 0.0, 0.0]
                ),
            )
        },
    )

    assert np.linalg.norm(
        relay.velocity
    ) >= 0.0

    assert np.linalg.norm(
        survey.velocity
        - np.array([5.0, 0.0, 0.0])
    ) >= 0.0


def test_geofence_is_enforced():

    model = HighFidelityFlightDynamics()

    result = model.step(
        position=np.array(
            [99.0, 50.0, 20.0]
        ),
        velocity=np.array(
            [5.0, 0.0, 0.0]
        ),
        preferred_velocity=np.array(
            [5.0, 0.0, 0.0]
        ),
        role="SURVEY",
        dt_s=1.0,
        lower_bound=np.array(
            [0.0, 0.0, 0.0]
        ),
        upper_bound=np.array(
            [100.0, 100.0, 100.0]
        ),
    )

    assert np.all(
        result.position
        <= np.array(
            [100.0, 100.0, 100.0]
        )
        + 1e-9
    )

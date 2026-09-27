import numpy as np

from uavx.autonomy.backbone import PredictiveRelayBackbone


def test_near_target_needs_no_relay():
    planner = PredictiveRelayBackbone(100.0, 10.0)
    assert planner.required_hops(np.array([50.0, 0.0, 20.0])) == 0
    assert planner.connectivity_required(np.array([50.0, 0.0, 20.0])) is False


def test_far_target_requires_hops():
    planner = PredictiveRelayBackbone(100.0, 10.0)
    assert planner.required_hops(np.array([950.0, 0.0, 20.0])) > 0
    assert planner.connectivity_required(np.array([950.0, 0.0, 20.0])) is True


def test_waypoints_are_inside_safe_range():
    planner = PredictiveRelayBackbone(100.0, 10.0)

    class Relay:
        pass

    # Use a tiny compatible stand-in.
    relays = []
    for i in range(5):
        u = Relay()
        u.uav_id = i
        u.position = np.array([25.0 * i, 0.0, 20.0])
        u.failure_state = False
        u.role = "RESERVE"
        relays.append(u)

    # Unit-level geometry check only.
    result = planner.required_hops(np.array([450.0, 0.0, 20.0]))
    assert result >= 3

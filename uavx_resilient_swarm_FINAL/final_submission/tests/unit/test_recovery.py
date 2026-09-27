from uavx.autonomy.recovery import (
    FailureClassification,
    FailureDetector,
    RecoveryCoordinator,
)


def test_failure_detector_requires_multiple_misses():
    d = FailureDetector(heartbeat_timeout_s=1.0, failure_threshold=3)
    d.observe(7, 0.0)

    assert d.update(2.0, set()) == []
    assert d.update(4.0, set()) == []
    found = d.update(6.0, set())

    assert len(found) == 1
    assert found[0].classification == FailureClassification.UAV_FAILURE


def test_recovery_selects_healthy_candidate():
    c = RecoveryCoordinator()
    action = c.select_replacement(3, [8, 9])

    assert action.replacement_uav_id == 8

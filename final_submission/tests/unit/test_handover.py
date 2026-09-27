from uavx.autonomy.handover import (
    EnergyPredictor,
    HandoverPhase,
    PriorityTaskManager,
    RelayHandover,
)


def test_energy_return_prediction():
    p = EnergyPredictor(1200.0, 0.15)

    assert p.remaining_time_s(1000.0) == 200.0
    assert p.should_return_home(1100.0, 30.0)


def test_handover_progression():
    h = RelayHandover(2, 5)
    h.advance()

    assert h.phase == HandoverPhase.RESERVE_DISPATCH


def test_priority_insertion():
    p = PriorityTaskManager()
    result = p.insert("emergency-1", 5.0, 1.0, [4, 9])

    assert result.selected_uav_id == 4

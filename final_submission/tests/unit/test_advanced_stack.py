import numpy as np

from uavx.autonomy import (
    AllocationRecord,
    CommunicationAwareAllocator,
    EnergyManager,
    FaultClassification,
    FaultEvidence,
    FailureDetector,
    GlobalPlanner,
    HandoverState,
    PriorityTaskInserter,
    RecoveryPlanner,
    RelayHandoverManager,
    ScenarioDefinition,
    ScenarioGenerator,
    AsymmetricCollisionAvoider,
    CommunicationPreservingPlanner,
)


def test_allocator_prefers_communication_feasible_candidate() -> None:
    allocator = CommunicationAwareAllocator()
    candidates = allocator.build_candidates(
        task_id=1,
        task_position=np.array([20.0, 0.0, 0.0]),
        uav_positions={
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([10.0, 0.0, 0.0]),
        },
        communication_margin={1: -0.1, 2: 0.8},
    )
    assert candidates[0].uav_id == 2


def test_partition_reconciliation_uses_newer_version() -> None:
    a = AllocationRecord(1, 1, 2.0, 1, 10.0)
    b = AllocationRecord(1, 2, 1.0, 2, 5.0)
    assert CommunicationAwareAllocator.reconcile_partition(a, b) == b


def test_priority_task_does_not_preempt_relay() -> None:
    decision = PriorityTaskInserter().decide(
        emergency_utility=100.0,
        current_utility=1.0,
        active_relays={7},
        candidate_uav=7,
    )
    assert not decision.accepted


def test_failure_detector_distinguishes_packet_loss() -> None:
    assessment = FailureDetector().assess(
        FaultEvidence(1.0, 0.95, 0.7, 0.8, 2)
    )
    assert assessment.classification == FaultClassification.PACKET_LOSS


def test_failure_detector_confirms_uav_failure() -> None:
    assessment = FailureDetector().assess(
        FaultEvidence(6.0, 0.0, 0.0, 0.5, 0)
    )
    assert assessment.classification == FaultClassification.UAV_FAILURE


def test_recovery_planner_selects_high_utility_candidate() -> None:
    planner = RecoveryPlanner()
    action = planner.plan(
        failed_uav_id=99,
        failed_position=np.array([0.0, 0.0, 0.0]),
        candidates={
            1: np.array([10.0, 0.0, 0.0]),
            2: np.array([50.0, 0.0, 0.0]),
        },
        battery_fraction={1: 0.9, 2: 0.2},
        link_margin={1: 0.9, 2: 0.1},
        task_burden={1: 0.1, 2: 0.0},
    )
    assert action.replacement_uav_id == 1


def test_energy_rth_threshold() -> None:
    estimate = EnergyManager(rth_threshold=0.25).estimate(0.20, 1200.0)
    assert estimate.should_rth
    assert estimate.estimated_remaining_s == 240.0


def test_energy_consumption_is_monotonic() -> None:
    e1 = EnergyManager.consume(1.0, 10.0, 1200.0)
    e2 = EnergyManager.consume(e1, 10.0, 1200.0)
    assert e2 < e1


def test_relay_handover_progression() -> None:
    manager = RelayHandoverManager()
    states = [manager.advance() for _ in range(3)]
    assert states == [
        HandoverState.RESERVE_DISPATCH,
        HandoverState.RESERVE_APPROACH,
        HandoverState.COMMUNICATION_OVERLAP,
    ]


def test_global_planner_respects_speed_limit() -> None:
    velocity = GlobalPlanner().velocity_toward(
        np.zeros(3),
        np.array([100.0, 0.0, 0.0]),
    )
    assert np.isclose(np.linalg.norm(velocity), 5.0)


def test_collision_avoidance_pushes_apart_close_uavs() -> None:
    avoider = AsymmetricCollisionAvoider()
    velocity = avoider.filter_velocity(
        np.array([0.0, 0.0, 0.0]),
        np.zeros(3),
        {2: np.array([10.0, 0.0, 0.0])},
        "SURVEY",
    )
    assert velocity[0] < 0.0


def test_communication_preserving_motion() -> None:
    planner = CommunicationPreservingPlanner(max_speed_mps=5.0, comm_range_m=100.0)
    result = planner.choose(
        current=np.array([0.0, 0.0, 0.0]),
        preferred=np.array([5.0, 0.0, 0.0]),
        anchor_positions=[np.array([90.0, 0.0, 0.0])],
    )
    assert result.preserves_connectivity


def test_scenario_seed_reproducibility() -> None:
    a = ScenarioGenerator(ScenarioDefinition(seed=42))
    b = ScenarioGenerator(ScenarioDefinition(seed=42))
    assert np.allclose(a.generate_pois(), b.generate_pois())


def test_scenario_seed_changes_output() -> None:
    a = ScenarioGenerator(ScenarioDefinition(seed=42))
    b = ScenarioGenerator(ScenarioDefinition(seed=43))
    assert not np.allclose(a.generate_pois(), b.generate_pois())

from __future__ import annotations

import json
from pathlib import Path

from uavx.autonomy.reporting import ReportingEngine
from uavx.autonomy.faults import (
    FailureDetector,
    FaultClassification,
    FaultEvidence,
)
from uavx.autonomy.recovery import RecoveryCoordinator
from uavx.autonomy.handover import (
    EnergyPredictor,
    HandoverPhase,
    PriorityTaskManager,
    RelayHandover,
)


def main() -> None:
    result = {}

    # Reporting.
    reporter = ReportingEngine(deadline_s=10.0)
    reporter.begin("task-1", 1, 100.0)
    report = reporter.transmit(
        "task-1",
        now_s=105.0,
        success=True,
        latency_ms=50.0,
    )

    result["reporting"] = {
        "status": report.status,
        "pdr": report.pdr,
        "latency_ms": report.latency_ms,
    }

    # Fault classification.
    detector = FailureDetector()

    packet_loss = detector.assess(
        FaultEvidence(1.0, 0.95, 0.70, 0.80, 2)
    )

    failure = detector.assess(
        FaultEvidence(6.0, 0.0, 0.0, 0.50, 0)
    )

    result["fault_detection"] = {
        "packet_loss": packet_loss.classification.value,
        "uav_failure": failure.classification.value,
        "failure_confirmed": failure.confirmed,
    }

    # Recovery.
    recovery = RecoveryCoordinator()
    action = recovery.select_replacement(
        failed_uav_id=3,
        candidate_ids=[8, 9],
    )

    result["recovery"] = {
        "failed_uav": action.failed_uav_id,
        "replacement_uav": action.replacement_uav_id,
        "reason": action.reason,
    }

    # Energy.
    energy = EnergyPredictor(
        endurance_s=1200.0,
        reserve_fraction=0.15,
    )

    result["energy"] = {
        "remaining_time_s": energy.remaining_time_s(1000.0),
        "rth_required": energy.should_return_home(1100.0, 30.0),
    }

    # Relay handover.
    handover = RelayHandover(
        old_relay_id=2,
        new_relay_id=5,
        started_at_s=500.0,
    )

    phases = [handover.phase.value]

    while handover.phase != HandoverPhase.OLD_RELAY_RTH:
        handover.advance()
        phases.append(handover.phase.value)

    result["handover"] = {
        "old_relay": handover.old_relay_id,
        "new_relay": handover.new_relay_id,
        "phases": phases,
    }

    # Priority task insertion.
    priority = PriorityTaskManager()
    insertion = priority.insert(
        task_id="emergency-1",
        priority=5.0,
        urgency=1.0,
        candidates=[4, 9],
    )

    result["priority_insertion"] = {
        "selected_uav": insertion.selected_uav_id,
        "priority": insertion.priority,
        "urgency": insertion.urgency,
    }

    out = Path("results/resilience_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()

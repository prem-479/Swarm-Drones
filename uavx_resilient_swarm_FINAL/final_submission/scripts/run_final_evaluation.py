from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavx.core.config import load_config
from uavx.core.models import TaskStatus
from uavx.simulation.engine import SimulationEngine


def main() -> None:
    config = load_config("configs/stage1.yaml")

    engine = SimulationEngine.from_config(config)
    state = engine.run()

    tasks = list(state.tasks.values())

    completed = sum(t.status == TaskStatus.COMPLETED for t in tasks)
    expired = sum(t.status == TaskStatus.EXPIRED for t in tasks)
    pending = sum(
        t.status in {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}
        for t in tasks
    )

    total = len(tasks)

    final_min_separation = float("inf")
    positions = [
        u.position.copy()
        for u in state.uavs.values()
        if not u.failure_state
    ]

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            final_min_separation = min(final_min_separation, distance)

    max_final_speed = max(
        (
            float(np.linalg.norm(u.velocity))
            for u in state.uavs.values()
            if not u.failure_state
        ),
        default=0.0,
    )

    swarm = getattr(engine, "swarm", None)
    stats = getattr(swarm, "stats", None)

    result = {
        "configuration": {
            "environment_m": [
                config.environment.width_m,
                config.environment.height_m,
            ],
            "uav_count": config.uav.count,
            "communication_range_m": config.communication.max_range_m,
            "endurance_s": config.uav.endurance_s,
            "mission_duration_s": config.simulation.mission_duration_s,
            "minimum_separation_m": config.uav.min_separation_m,
            "maximum_speed_mps": config.uav.max_speed_mps,
            "maximum_altitude_m": config.environment.max_altitude_m,
            "poi_count": config.scenario.poi_count,
            "reporting_deadline_s": config.mission.poi_reporting_deadline_s,
        },
        "simulation": {
            "simulated_time_s": state.timestamp_s,
            "status": state.mission.status.value,
        },
        "mission": {
            "tasks": total,
            "completed": completed,
            "expired": expired,
            "pending_or_active": pending,
            "completion_rate": completed / total if total else 0.0,
        },
        "swarm": {
            "role_changes": getattr(stats, "role_changes", 0),
            "relay_reallocations": getattr(stats, "relay_reallocations", 0),
            "recoveries": getattr(stats, "recoveries", 0),
            "rth_events": getattr(stats, "rth_events", 0),
        },
        "safety": {
            "final_pairwise_minimum_separation_m": final_min_separation,
            "required_minimum_separation_m": config.uav.min_separation_m,
            "maximum_final_speed_mps": max_final_speed,
            "speed_limit_mps": config.uav.max_speed_mps,
        },
        "network": {
            "nodes": len(state.network.nodes),
            "edges": len(state.network.edges),
        },
    }

    output = Path("results/final_evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()

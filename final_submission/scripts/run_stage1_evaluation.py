from __future__ import annotations

import json
from pathlib import Path

from uavx.core.config import load_config
from uavx.simulation.engine import SimulationEngine
from uavx.autonomy.metrics import MissionMetrics, completion_rate


CONFIG = Path("configs/stage1.yaml")
OUT = Path("results/stage1_metrics.json")


def main() -> None:
    config = load_config(CONFIG)
    engine = SimulationEngine.from_config(config)

    while engine.state.timestamp_s < config.simulation.mission_duration_s:
        engine.step(config.simulation.timestep_s)

    tasks = list(engine.state.tasks.values())
    completed = sum(
        1 for t in tasks if getattr(t.status, "value", t.status) == "COMPLETED"
    )

    metrics = MissionMetrics(
        mission_completion_rate=completion_rate(completed, len(tasks)),
        relay_reallocations=engine.swarm.stats.relay_reallocations,
        role_changes=engine.swarm.stats.role_changes,
        recoveries=engine.swarm.stats.recoveries,
        rth_events=engine.swarm.stats.rth_events,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.save(str(OUT))

    print(json.dumps(metrics.to_dict(), indent=2))
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()

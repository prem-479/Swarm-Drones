from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from uavx.core.config import load_config
from uavx.core.models import TaskStatus
from uavx.simulation.engine import SimulationEngine


def run(seed: int, base):
    cfg = replace(
        base,
        simulation=replace(base.simulation, seed=seed),
    )

    engine = SimulationEngine.from_config(cfg)
    state = engine.run()

    tasks = list(state.tasks.values())
    completed = sum(t.status == TaskStatus.COMPLETED for t in tasks)

    return {
        "seed": seed,
        "tasks": len(tasks),
        "completed": completed,
        "completion_rate": completed / len(tasks) if tasks else 0.0,
    }


def main() -> None:
    base = load_config("configs/stage1.yaml")
    seeds = list(range(10))

    runs = [run(seed, base) for seed in seeds]

    result = {
        "experiment": "stage1_seed_sweep",
        "seeds": seeds,
        "runs": runs,
        "mean_completion_rate": (
            sum(r["completion_rate"] for r in runs) / len(runs)
        ),
    }

    output = Path("results/monte_carlo_summary.json")
    output.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()

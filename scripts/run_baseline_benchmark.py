from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uavx.autonomy.backbone import PredictiveRelayBackbone


def main() -> None:
    planner = PredictiveRelayBackbone(
        max_range_m=100.0,
        safety_margin_m=10.0,
    )

    distances = [50, 100, 250, 500, 750, 950]

    rows = []

    for distance in distances:
        target = np.array(
            [float(distance), 0.0, 20.0],
            dtype=float,
        )

        rows.append(
            {
                "target_distance_m": distance,
                "reactive_relay_count": 0,
                "predictive_required_relays": planner.required_hops(target),
            }
        )

    result = {
        "benchmark": "relay_geometry",
        "rows": rows,
    }

    output = Path("results/baseline_benchmark.json")
    output.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()

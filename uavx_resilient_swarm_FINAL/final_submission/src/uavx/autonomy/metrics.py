from __future__ import annotations

from dataclasses import dataclass, asdict
import json


@dataclass
class MissionMetrics:
    mission_completion_rate: float = 0.0
    mean_completion_latency_s: float = 0.0
    communication_pdr: float = 0.0
    communication_latency_ms: float = 0.0
    connectivity_downtime_s: float = 0.0
    relay_reallocations: int = 0
    role_changes: int = 0
    failures_detected: int = 0
    recoveries: int = 0
    task_reassignments: int = 0
    minimum_separation_m: float = 0.0
    maximum_speed_mps: float = 0.0
    rth_events: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def completion_rate(completed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return completed / total


def normalized(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))

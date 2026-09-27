from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from uavx.communication import GCS_NODE_ID
from uavx.core.models import UAVRole, UAVState


@dataclass(frozen=True)
class RelayWaypoint:
    uav_id: int
    position: np.ndarray
    hop_index: int
    distance_to_gcs: float


class PredictiveRelayBackbone:
    """
    Lightweight predictive relay-chain planner.

    It plans relay waypoints before a survey UAV leaves the current
    communication region. It does not use future failure knowledge.
    """

    def __init__(
        self,
        max_range_m: float,
        safety_margin_m: float = 10.0,
    ) -> None:
        if max_range_m <= 0:
            raise ValueError("max_range_m must be > 0")

        self.max_range_m = float(max_range_m)
        self.safety_margin_m = float(
            np.clip(safety_margin_m, 1.0, max_range_m * 0.45)
        )

    @property
    def hop_spacing_m(self) -> float:
        return self.max_range_m - self.safety_margin_m

    def required_hops(self, target: np.ndarray) -> int:
        distance = float(np.linalg.norm(target[:2]))
        return max(0, int(np.ceil(distance / self.hop_spacing_m)) - 1)

    def plan(
        self,
        target: np.ndarray,
        relay_uavs: list[UAVState],
    ) -> list[RelayWaypoint]:
        target_xy = np.asarray(target, dtype=float).copy()
        target_xy[2] = target[2]

        distance = float(np.linalg.norm(target_xy[:2]))
        if distance <= self.hop_spacing_m:
            return []

        direction = np.zeros(3, dtype=float)
        direction[:2] = target_xy[:2] / distance
        direction[2] = 0.0

        needed = self.required_hops(target_xy)

        available = [
            u
            for u in relay_uavs
            if not u.failure_state
            and u.role not in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }
        ]

        available.sort(
            key=lambda u: (
                float(np.linalg.norm(u.position[:2])),
                u.uav_id,
            )
        )

        waypoints: list[RelayWaypoint] = []

        # Place relays between GCS and target at safe spacing.
        for hop in range(1, min(needed, len(available)) + 1):
            relay = available[hop - 1]
            point = direction * (
                self.hop_spacing_m * hop
            )
            point[2] = target_xy[2]

            waypoints.append(
                RelayWaypoint(
                    uav_id=relay.uav_id,
                    position=point,
                    hop_index=hop,
                    distance_to_gcs=float(np.linalg.norm(point[:2])),
                )
            )

        return waypoints

    def connectivity_required(self, target: np.ndarray) -> bool:
        return float(np.linalg.norm(target[:2])) > self.hop_spacing_m

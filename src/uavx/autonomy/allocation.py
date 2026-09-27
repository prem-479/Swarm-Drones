from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Dict, Iterable, Mapping, Optional
import numpy as np


@dataclass(frozen=True)
class AllocationWeights:
    task_value: float = 1.0
    priority: float = 2.0
    travel: float = 1.0
    communication: float = 2.0
    link_margin: float = 1.0
    relay_burden: float = 1.0
    energy: float = 1.0
    urgency: float = 1.0


@dataclass(frozen=True)
class AllocationCandidate:
    uav_id: int
    task_id: int
    utility: float
    travel_time_s: float
    communication_feasible: bool
    link_margin: float
    requires_relay: bool


@dataclass(frozen=True)
class AllocationRecord:
    task_id: int
    winner: int
    bid: float
    version: int
    timestamp_s: float
    status: str = "ACTIVE"


@dataclass(frozen=True)
class PreemptionDecision:
    accepted: bool
    reason: str
    utility_gain: float = 0.0


class CommunicationAwareAllocator:
    def __init__(self, weights: AllocationWeights | None = None) -> None:
        self.weights = weights or AllocationWeights()

    def build_candidates(
        self,
        task_id: int,
        task_position: np.ndarray,
        uav_positions: Mapping[int, np.ndarray],
        active_relays: Iterable[int] = (),
        max_speed_mps: float = 5.0,
        communication_margin: Optional[Mapping[int, float]] = None,
        task_priority: float = 1.0,
        urgency: float = 1.0,
    ) -> list[AllocationCandidate]:
        relay_set = set(active_relays)
        margins = communication_margin or {}
        result: list[AllocationCandidate] = []

        for uav_id, position in uav_positions.items():
            distance = float(np.linalg.norm(np.asarray(position) - task_position))
            travel_time = distance / max(max_speed_mps, 1e-9)
            margin = float(margins.get(uav_id, 1.0))
            feasible = margin >= 0.0
            requires_relay = uav_id not in relay_set and margin < 0.25

            utility = (
                self.weights.task_value
                + self.weights.priority * task_priority
                + self.weights.urgency * urgency
                - self.weights.travel * travel_time
                + self.weights.communication * float(feasible)
                + self.weights.link_margin * margin
                - self.weights.relay_burden * float(requires_relay)
                + self.weights.energy * 1.0
            )

            result.append(
                AllocationCandidate(
                    uav_id=uav_id,
                    task_id=task_id,
                    utility=utility,
                    travel_time_s=travel_time,
                    communication_feasible=feasible,
                    link_margin=margin,
                    requires_relay=requires_relay,
                )
            )

        return sorted(result, key=lambda c: (-c.utility, c.uav_id))

    def allocate(
        self,
        tasks: Iterable[tuple[int, np.ndarray]],
        uav_positions: Mapping[int, np.ndarray],
        active_relays: Iterable[int] = (),
        max_speed_mps: float = 5.0,
        communication_margin: Optional[Mapping[int, float]] = None,
    ) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        used: set[int] = set()

        for task_id, task_position in sorted(tasks, key=lambda x: x[0]):
            candidates = self.build_candidates(
                task_id,
                task_position,
                uav_positions,
                active_relays,
                max_speed_mps,
                communication_margin,
            )
            for candidate in candidates:
                if candidate.uav_id not in used and candidate.communication_feasible:
                    assignments[task_id] = candidate.uav_id
                    used.add(candidate.uav_id)
                    break

        return assignments

    @staticmethod
    def reconcile_partition(
        local: AllocationRecord,
        remote: AllocationRecord,
    ) -> AllocationRecord:
        key_local = (local.version, local.timestamp_s, local.bid, -local.winner)
        key_remote = (remote.version, remote.timestamp_s, remote.bid, -remote.winner)
        return local if key_local >= key_remote else remote


class PriorityTaskInserter:
    def __init__(self, allocator: CommunicationAwareAllocator | None = None) -> None:
        self.allocator = allocator or CommunicationAwareAllocator()

    def decide(
        self,
        emergency_utility: float,
        current_utility: float,
        active_relays: set[int],
        candidate_uav: int,
        protect_relays: bool = True,
    ) -> PreemptionDecision:
        if protect_relays and candidate_uav in active_relays:
            return PreemptionDecision(False, "active relay protected", 0.0)

        gain = emergency_utility - current_utility
        if gain > 0.0:
            return PreemptionDecision(True, "positive utility gain", gain)

        return PreemptionDecision(False, "no positive utility gain", gain)

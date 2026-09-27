from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HandoverPhase(str, Enum):
    PRE_HANDOVER = "PRE_HANDOVER"
    RESERVE_DISPATCH = "RESERVE_DISPATCH"
    RESERVE_APPROACH = "RESERVE_APPROACH"
    COMMUNICATION_OVERLAP = "COMMUNICATION_OVERLAP"
    ROLE_TRANSFER = "ROLE_TRANSFER"
    OLD_RELAY_RTH = "OLD_RELAY_RTH"


@dataclass
class RelayHandover:
    old_relay_id: int
    new_relay_id: int
    phase: HandoverPhase = HandoverPhase.PRE_HANDOVER
    started_at_s: float = 0.0
    completed_at_s: float | None = None

    def advance(self) -> None:
        order = list(HandoverPhase)
        index = order.index(self.phase)

        if index + 1 < len(order):
            self.phase = order[index + 1]
        else:
            self.completed_at_s = self.started_at_s


class EnergyPredictor:
    def __init__(
        self,
        endurance_s: float,
        reserve_fraction: float,
    ) -> None:
        self.endurance_s = endurance_s
        self.reserve_fraction = reserve_fraction

    def remaining_time_s(self, flight_time_s: float) -> float:
        return max(0.0, self.endurance_s - flight_time_s)

    def should_return_home(
        self,
        flight_time_s: float,
        estimated_home_time_s: float,
    ) -> bool:
        remaining = self.remaining_time_s(flight_time_s)
        reserve = self.endurance_s * self.reserve_fraction

        return remaining <= estimated_home_time_s + reserve


@dataclass(frozen=True)
class PriorityInsertion:
    task_id: str
    priority: float
    urgency: float
    selected_uav_id: int | None
    displaced_task_id: str | None


class PriorityTaskManager:
    def insert(
        self,
        task_id: str,
        priority: float,
        urgency: float,
        candidates: list[int],
    ) -> PriorityInsertion:
        selected = min(candidates) if candidates else None

        return PriorityInsertion(
            task_id=task_id,
            priority=priority,
            urgency=urgency,
            selected_uav_id=selected,
            displaced_task_id=None,
        )

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HandoverState(str, Enum):
    PRE_HANDOVER = "pre_handover"
    RESERVE_DISPATCH = "reserve_dispatch"
    RESERVE_APPROACH = "reserve_approach"
    COMMUNICATION_OVERLAP = "communication_overlap"
    ROLE_TRANSFER = "role_transfer"
    OLD_RELAY_RTH = "old_relay_rth"
    COMPLETE = "complete"


@dataclass(frozen=True)
class EnergyEstimate:
    battery_fraction: float
    estimated_remaining_s: float
    should_rth: bool


class EnergyManager:
    def __init__(self, rth_threshold: float = 0.25, reserve_fraction: float = 0.15) -> None:
        self.rth_threshold = rth_threshold
        self.reserve_fraction = reserve_fraction

    def estimate(self, battery_fraction: float, endurance_s: float) -> EnergyEstimate:
        fraction = max(0.0, min(1.0, battery_fraction))
        remaining = fraction * endurance_s
        return EnergyEstimate(
            fraction,
            remaining,
            fraction <= self.rth_threshold,
        )

    @staticmethod
    def consume(
        battery_fraction: float,
        dt_s: float,
        endurance_s: float,
    ) -> float:
        return max(0.0, battery_fraction - dt_s / max(endurance_s, 1e-9))


class RelayHandoverManager:
    ORDER = (
        HandoverState.PRE_HANDOVER,
        HandoverState.RESERVE_DISPATCH,
        HandoverState.RESERVE_APPROACH,
        HandoverState.COMMUNICATION_OVERLAP,
        HandoverState.ROLE_TRANSFER,
        HandoverState.OLD_RELAY_RTH,
        HandoverState.COMPLETE,
    )

    def __init__(self) -> None:
        self.state = HandoverState.PRE_HANDOVER

    def advance(self, communication_overlap: bool = False) -> HandoverState:
        if self.state == HandoverState.COMMUNICATION_OVERLAP and not communication_overlap:
            return self.state

        index = self.ORDER.index(self.state)
        if index < len(self.ORDER) - 1:
            self.state = self.ORDER[index + 1]

        return self.state

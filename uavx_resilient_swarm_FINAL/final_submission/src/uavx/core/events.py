from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import heapq


class EventType(str, Enum):
    POI_SPAWN = "POI_SPAWN"
    POI_DISCOVERED = "POI_DISCOVERED"
    TASK_ASSIGNED = "TASK_ASSIGNED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_REASSIGNED = "TASK_REASSIGNED"
    LINK_DEGRADED = "LINK_DEGRADED"
    LINK_FAILED = "LINK_FAILED"
    UAV_FAILED = "UAV_FAILED"
    BATTERY_WARNING = "BATTERY_WARNING"
    RTH_TRIGGERED = "RTH_TRIGGERED"
    RELAY_HANDOVER = "RELAY_HANDOVER"
    NETWORK_RECONFIGURATION = "NETWORK_RECONFIGURATION"
    HIGH_PRIORITY_TASK = "HIGH_PRIORITY_TASK"
    MISSION_END = "MISSION_END"


@dataclass(order=True)
class SimulationEvent:
    timestamp_s: float
    sequence: int
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )


class EventQueue:
    """Priority queue for discrete simulation events."""

    def __init__(self) -> None:
        self._queue: list[SimulationEvent] = []
        self._sequence = 0

    def push(
        self,
        timestamp_s: float,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        if timestamp_s < 0:
            raise ValueError("Event timestamp cannot be negative")

        event = SimulationEvent(
            timestamp_s=timestamp_s,
            sequence=self._sequence,
            event_type=event_type,
            payload=payload or {},
        )

        self._sequence += 1
        heapq.heappush(self._queue, event)
        return event

    def peek(self) -> SimulationEvent | None:
        return self._queue[0] if self._queue else None

    def pop(self) -> SimulationEvent | None:
        if not self._queue:
            return None
        return heapq.heappop(self._queue)

    def pop_ready(self, timestamp_s: float) -> list[SimulationEvent]:
        ready: list[SimulationEvent] = []

        while self._queue and self._queue[0].timestamp_s <= timestamp_s:
            ready.append(heapq.heappop(self._queue))

        return ready

    def __len__(self) -> int:
        return len(self._queue)

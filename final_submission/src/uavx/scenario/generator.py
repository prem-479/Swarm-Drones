from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from uavx.core.events import EventQueue, EventType
from uavx.core.models import PoI, TaskStatus


@dataclass
class ScenarioGenerator:
    width_m: float
    height_m: float
    max_pois: int
    reporting_deadline_s: float
    seed: int

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("Scenario bounds must be positive")

        if self.max_pois < 0:
            raise ValueError("max_pois cannot be negative")

        self.rng = np.random.default_rng(self.seed)
        self._next_poi_id = 0

    def create_poi(self, timestamp_s: float) -> PoI:
        """Create one reproducible random PoI."""
        position = np.array(
            [
                self.rng.uniform(0.0, self.width_m),
                self.rng.uniform(0.0, self.height_m),
                0.0,
            ],
            dtype=float,
        )

        priority = float(self.rng.uniform(1.0, 5.0))

        poi = PoI(
            poi_id=self._next_poi_id,
            position=position,
            priority=priority,
            spawn_time_s=timestamp_s,
            discovery_time_s=None,
            status=TaskStatus.PENDING,
            reporting_deadline_s=self.reporting_deadline_s,
        )

        self._next_poi_id += 1
        return poi

    def schedule_poi_events(
        self,
        event_queue: EventQueue,
        mission_duration_s: float,
    ) -> int:
        """Schedule the configured number of stochastic PoI spawns."""
        if mission_duration_s <= 0:
            raise ValueError("mission_duration_s must be positive")

        if self.max_pois == 0:
            return 0

        spawn_times = np.sort(
            self.rng.uniform(
                low=0.0,
                high=mission_duration_s,
                size=self.max_pois,
            )
        )

        for timestamp_s in spawn_times:
            event_queue.push(
                timestamp_s=float(timestamp_s),
                event_type=EventType.POI_SPAWN,
            )

        return self.max_pois

    def iter_random_pois(
        self,
        count: int,
        timestamp_s: float = 0.0,
    ) -> Iterator[PoI]:
        for _ in range(count):
            yield self.create_poi(timestamp_s)

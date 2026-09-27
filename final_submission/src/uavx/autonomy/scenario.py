from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass(frozen=True)
class ScenarioEvent:
    time_s: float
    event_type: str
    target_id: int | None = None
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioDefinition:
    seed: int
    width_m: float = 1000.0
    height_m: float = 1000.0
    max_altitude_m: float = 100.0
    poi_count: int = 10
    mission_duration_s: float = 2700.0


class ScenarioGenerator:
    def __init__(self, definition: ScenarioDefinition) -> None:
        self.definition = definition
        self.rng = np.random.default_rng(definition.seed)

    def generate_pois(self) -> np.ndarray:
        count = self.definition.poi_count
        xy = self.rng.uniform(
            low=[0.0, 0.0],
            high=[self.definition.width_m, self.definition.height_m],
            size=(count, 2),
        )
        z = np.zeros((count, 1))
        return np.hstack([xy, z])

    def generate_events(self) -> List[ScenarioEvent]:
        events: list[ScenarioEvent] = []
        for poi_id in range(self.definition.poi_count):
            spawn = float(
                self.rng.uniform(
                    0.0,
                    max(1.0, self.definition.mission_duration_s - 10.0),
                )
            )
            events.append(
                ScenarioEvent(
                    time_s=spawn,
                    event_type="poi_spawn",
                    target_id=poi_id,
                )
            )

        return sorted(events, key=lambda event: (event.time_s, event.target_id or -1))

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class MotionLimits:
    max_speed_mps: float = 5.0
    min_separation_m: float = 20.0


@dataclass(frozen=True)
class MotionCandidate:
    velocity: np.ndarray
    score: float
    safe: bool
    preserves_connectivity: bool


class GlobalPlanner:
    def __init__(self, limits: MotionLimits | None = None) -> None:
        self.limits = limits or MotionLimits()

    def velocity_toward(
        self,
        current: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        delta = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            return np.zeros(3)

        return delta / distance * min(distance, self.limits.max_speed_mps)


class GeofenceGuard:
    def __init__(self, lower: Sequence[float], upper: Sequence[float]) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

    def clamp_position(self, position: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(position, dtype=float), self.lower, self.upper)

    def clamp_velocity(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        result = np.asarray(velocity, dtype=float).copy()
        pos = np.asarray(position, dtype=float)
        for axis in range(3):
            if pos[axis] <= self.lower[axis] and result[axis] < 0:
                result[axis] = 0.0
            if pos[axis] >= self.upper[axis] and result[axis] > 0:
                result[axis] = 0.0
        return result


class AsymmetricCollisionAvoider:
    def __init__(
        self,
        min_separation_m: float = 20.0,
        relay_weight: float = 0.25,
        survey_weight: float = 1.0,
    ) -> None:
        self.min_separation_m = min_separation_m
        self.relay_weight = relay_weight
        self.survey_weight = survey_weight

    def filter_velocity(
        self,
        uav_position: np.ndarray,
        desired_velocity: np.ndarray,
        neighbors: Mapping[int, np.ndarray],
        uav_role: str = "SURVEY",
    ) -> np.ndarray:
        velocity = np.asarray(desired_velocity, dtype=float).copy()
        responsibility = (
            self.relay_weight if uav_role.upper() == "RELAY"
            else self.survey_weight
        )

        for neighbor_position in neighbors.values():
            delta = np.asarray(uav_position) - np.asarray(neighbor_position)
            distance = float(np.linalg.norm(delta))
            if 0.0 < distance < self.min_separation_m:
                away = delta / distance
                velocity += responsibility * (
                    (self.min_separation_m - distance) / self.min_separation_m
                ) * away

        speed = float(np.linalg.norm(velocity))
        if speed > 5.0:
            velocity = velocity / speed * 5.0

        return velocity


class CommunicationPreservingPlanner:
    def __init__(self, max_speed_mps: float = 5.0, comm_range_m: float = 100.0) -> None:
        self.max_speed_mps = max_speed_mps
        self.comm_range_m = comm_range_m

    def choose(
        self,
        current: np.ndarray,
        preferred: np.ndarray,
        anchor_positions: Sequence[np.ndarray],
    ) -> MotionCandidate:
        preferred = np.asarray(preferred, dtype=float)
        speed = float(np.linalg.norm(preferred))
        if speed > self.max_speed_mps:
            preferred = preferred / speed * self.max_speed_mps

        projected = np.asarray(current, dtype=float) + preferred
        connectivity = True

        if anchor_positions:
            connectivity = any(
                np.linalg.norm(projected - np.asarray(anchor)) <= self.comm_range_m
                for anchor in anchor_positions
            )

        return MotionCandidate(
            velocity=preferred,
            score=float(np.linalg.norm(preferred)),
            safe=True,
            preserves_connectivity=connectivity,
        )

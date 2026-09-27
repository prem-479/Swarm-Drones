from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from uavx.core.models import UAVRole


EPS = 1e-9


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n <= EPS:
        return np.zeros(3, dtype=float)
    return np.asarray(v, dtype=float) / n


def _rotate_xy(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))

    x = float(v[0])
    y = float(v[1])

    out = np.asarray(v, dtype=float).copy()
    out[0] = c * x - s * y
    out[1] = s * x + c * y
    return out


@dataclass(frozen=True)
class RectObstacle:
    """
    Optional static no-fly rectangle for stress testing.

    Official Stage-1 does not define static obstacles. This class is
    therefore an optional simulator stress-test feature, not part of
    the official baseline scoring configuration.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(
        self,
        position: np.ndarray,
        margin: float = 0.0,
    ) -> bool:
        x = float(position[0])
        y = float(position[1])

        return (
            self.x_min - margin <= x <= self.x_max + margin
            and self.y_min - margin <= y <= self.y_max + margin
        )

    def segment_hits(
        self,
        start: np.ndarray,
        end: np.ndarray,
        margin: float = 0.0,
    ) -> bool:
        distance = _norm(
            np.asarray(end, dtype=float)
            - np.asarray(start, dtype=float)
        )

        samples = max(
            2,
            int(np.ceil(distance / 5.0)),
        )

        for alpha in np.linspace(
            0.0,
            1.0,
            samples + 1,
        ):
            point = (
                np.asarray(start, dtype=float) * (1.0 - alpha)
                + np.asarray(end, dtype=float) * alpha
            )

            if self.contains(point, margin):
                return True

        return False


@dataclass
class MotionDiagnostics:
    min_separation_m: float = float("inf")
    collision_interventions: int = 0
    obstacle_interventions: int = 0
    geofence_interventions: int = 0
    max_speed_mps: float = 0.0


class ProfessionalMotionSystem:
    """
    Professional 3D point-mass swarm motion layer.

    Design properties:

    * simultaneous velocity planning from a common state snapshot
    * predictive collision avoidance
    * role-dependent maneuver responsibility
    * acceleration-limited flight
    * braking near targets
    * hard 20 m minimum separation
    * geofence protection
    * optional static no-fly rectangles
    * deterministic behavior under a fixed simulation seed
    """

    def __init__(
        self,
        max_speed_mps: float,
        max_acceleration_mps2: float,
        min_separation_m: float,
        width_m: float,
        height_m: float,
        max_altitude_m: float,
        reserve_slots: Mapping[int, np.ndarray] | None = None,
    ) -> None:

        self.max_speed_mps = float(max_speed_mps)
        self.max_acceleration_mps2 = float(
            max_acceleration_mps2
        )
        self.min_separation_m = float(min_separation_m)

        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.max_altitude_m = float(max_altitude_m)

        # Predictive horizon is long enough to see a 20 m
        # separation breach at the official 5 m/s maximum speed.
        self.prediction_horizon_s = 4.0

        # Safety margin above the official 20 m boundary.
        self.separation_buffer_m = 1.0

        # Soft interaction zone. UAVs begin maneuvering before
        # reaching the hard boundary.
        self.soft_radius_m = max(
            self.min_separation_m + 12.0,
            32.0,
        )

        self.obstacle_clearance_m = 6.0

        self.obstacles: list[RectObstacle] = []

        self.reserve_slots = {
            int(uid): np.asarray(
                position,
                dtype=float,
            ).copy()
            for uid, position
            in (reserve_slots or {}).items()
        }

        self.diagnostics = MotionDiagnostics()

    # ------------------------------------------------------------------
    # Optional static obstacle layer
    # ------------------------------------------------------------------

    def set_obstacles(
        self,
        obstacles: list[RectObstacle],
    ) -> None:

        self.obstacles = list(obstacles)

    # ------------------------------------------------------------------
    # Role-dependent responsibility
    # ------------------------------------------------------------------

    def maneuver_weight(
        self,
        role: UAVRole,
    ) -> float:

        if role == UAVRole.RELAY:
            # Relays are connectivity-critical and should move least.
            return 4.0

        if role == UAVRole.SURVEY:
            # Survey UAVs absorb most lateral avoidance.
            return 1.0

        if role == UAVRole.RESERVE:
            return 2.0

        if role in {
            UAVRole.RECOVERY,
            UAVRole.TRANSIT,
        }:
            return 1.25

        if role == UAVRole.RETURN_HOME:
            return 1.5

        return 2.0

    # ------------------------------------------------------------------
    # Preferred velocity
    # ------------------------------------------------------------------

    def preferred_velocity(
        self,
        uav,
    ) -> np.ndarray:

        position = np.asarray(
            uav.position,
            dtype=float,
        )

        target = getattr(
            uav,
            "target",
            None,
        )

        role = getattr(
            uav,
            "role",
            UAVRole.RESERVE,
        )

        # Landing / failed UAVs remain stationary.
        if role in {
            UAVRole.FAILED,
            UAVRole.LANDING,
        }:

            return np.zeros(3, dtype=float)

        # RTH uses home position.
        if (
            role == UAVRole.RETURN_HOME
            and target is None
        ):

            target = np.asarray(
                uav.home_position,
                dtype=float,
            )

        # Reserve UAVs hold a stable formation slot.
        if (
            target is None
            and role == UAVRole.RESERVE
            and uav.uav_id in self.reserve_slots
        ):

            target = self.reserve_slots[
                uav.uav_id
            ]

        if target is None:

            # Relays without an explicit controller target should
            # station-keep rather than drift.
            return np.zeros(3, dtype=float)

        target = np.asarray(
            target,
            dtype=float,
        ).copy()

        error = target - position
        distance = _norm(error)

        if distance <= 0.5:
            return np.zeros(3, dtype=float)

        direction = _unit(error)

        # Braking law:
        #
        # v <= sqrt(2 a d)
        #
        # prevents the common "fly through the waypoint, then turn
        # back" oscillation.
        braking_speed = np.sqrt(
            max(
                0.0,
                2.0
                * self.max_acceleration_mps2
                * max(distance - 0.5, 0.0),
            )
        )

        desired_speed = min(
            self.max_speed_mps,
            braking_speed,
        )

        return direction * desired_speed

    # ------------------------------------------------------------------
    # Predictive pairwise safety
    # ------------------------------------------------------------------

    def _predicted_min_separation(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        neighbor,
        horizon_s: float,
    ) -> float:

        relative_position = (
            np.asarray(neighbor.position, dtype=float)
            - np.asarray(position, dtype=float)
        )

        relative_velocity = (
            np.asarray(velocity, dtype=float)
            - np.asarray(neighbor.velocity, dtype=float)
        )

        vv = float(
            np.dot(
                relative_velocity,
                relative_velocity,
            )
        )

        if vv <= EPS:

            return _norm(relative_position)

        t_star = -float(
            np.dot(
                relative_position,
                relative_velocity,
            )
        ) / vv

        t_star = float(
            np.clip(
                t_star,
                0.0,
                horizon_s,
            )
        )

        closest = (
            relative_position
            + relative_velocity * t_star
        )

        return _norm(closest)

    def _candidate_safe(
        self,
        uav,
        candidate: np.ndarray,
        neighbors,
    ) -> tuple[bool, float]:

        min_predicted = float("inf")

        required = (
            self.min_separation_m
            + self.separation_buffer_m
        )

        for neighbor in neighbors.values():

            if neighbor.uav_id == uav.uav_id:
                continue

            predicted = self._predicted_min_separation(
                uav.position,
                candidate,
                neighbor,
                self.prediction_horizon_s,
            )

            min_predicted = min(
                min_predicted,
                predicted,
            )

            if predicted < required:
                return False, min_predicted

        # Static obstacle prediction.
        start = np.asarray(
            uav.position,
            dtype=float,
        )

        for sample_t in np.linspace(
            0.2,
            self.prediction_horizon_s,
            14,
        ):

            end = start + candidate * sample_t

            # Geofence.
            if (
                end[0] < 0.0
                or end[0] > self.width_m
                or end[1] < 0.0
                or end[1] > self.height_m
                or end[2] < 0.0
                or end[2] > self.max_altitude_m
            ):

                return False, min_predicted

            for obstacle in self.obstacles:

                if obstacle.segment_hits(
                    start,
                    end,
                    self.obstacle_clearance_m,
                ):

                    return False, min_predicted

        return True, min_predicted

    # ------------------------------------------------------------------
    # Candidate velocity generation
    # ------------------------------------------------------------------

    def _candidate_velocities(
        self,
        preferred: np.ndarray,
    ) -> list[np.ndarray]:

        speed = _norm(preferred)

        if speed <= 0.05:

            speeds = [
                0.0,
                0.5 * self.max_speed_mps,
            ]

            base = np.array(
                [1.0, 0.0, 0.0],
                dtype=float,
            )

        else:

            speeds = [
                speed,
                min(
                    self.max_speed_mps,
                    0.85 * speed,
                ),
                min(
                    self.max_speed_mps,
                    0.65 * speed,
                ),
                min(
                    self.max_speed_mps,
                    0.40 * speed,
                ),
            ]

            base = preferred.copy()

        angles = [
            0.0,
            np.deg2rad(15.0),
            np.deg2rad(-15.0),
            np.deg2rad(30.0),
            np.deg2rad(-30.0),
            np.deg2rad(45.0),
            np.deg2rad(-45.0),
            np.deg2rad(60.0),
            np.deg2rad(-60.0),
            np.deg2rad(90.0),
            np.deg2rad(-90.0),
            np.deg2rad(120.0),
            np.deg2rad(-120.0),
            np.pi,
        ]

        candidates: list[np.ndarray] = []

        for magnitude in speeds:

            for angle in angles:

                if magnitude <= EPS:

                    candidates.append(
                        np.zeros(3, dtype=float)
                    )
                    continue

                rotated = _rotate_xy(
                    base,
                    angle,
                )

                horizontal_norm = _norm(
                    rotated[:2]
                )

                if horizontal_norm <= EPS:

                    direction = _unit(
                        rotated
                    )

                else:

                    horizontal = (
                        rotated[:2]
                        / horizontal_norm
                    )

                    # Keep some vertical intent from the
                    # original preferred velocity.
                    vertical = (
                        preferred[2]
                        / max(
                            _norm(preferred),
                            self.max_speed_mps,
                        )
                    )

                    direction = np.array(
                        [
                            horizontal[0],
                            horizontal[1],
                            vertical,
                        ],
                        dtype=float,
                    )

                    direction = _unit(
                        direction
                    )

                candidates.append(
                    direction * magnitude
                )

        # Deterministic de-duplication.
        result = []
        seen = set()

        for candidate in candidates:

            key = tuple(
                np.round(
                    candidate,
                    4,
                )
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(candidate)

        return result

    # ------------------------------------------------------------------
    # Velocity selection
    # ------------------------------------------------------------------

    def safe_velocity(
        self,
        uav,
        preferred: np.ndarray,
        neighbors,
    ) -> np.ndarray:

        candidates = self._candidate_velocities(
            preferred
        )

        role_weight = self.maneuver_weight(
            getattr(
                uav,
                "role",
                UAVRole.RESERVE,
            )
        )

        best = None
        best_score = float("inf")

        target = getattr(
            uav,
            "target",
            None,
        )

        target_direction = np.zeros(
            3,
            dtype=float,
        )

        if target is not None:

            target_direction = _unit(
                np.asarray(target, dtype=float)
                - np.asarray(uav.position, dtype=float)
            )

        for candidate in candidates:

            safe, predicted_sep = (
                self._candidate_safe(
                    uav,
                    candidate,
                    neighbors,
                )
            )

            deviation = _norm(
                candidate - preferred
            )

            # Larger predicted separation is preferable.
            separation_reward = (
                1.0
                / max(
                    predicted_sep,
                    1.0,
                )
            )

            progress = -float(
                np.dot(
                    candidate,
                    target_direction,
                )
            )

            score = (
                role_weight
                * deviation
                + 3.0 * separation_reward
                + progress
            )

            # Unsafe candidates get a very large penalty,
            # but remain available as a last-resort option.
            if not safe:
                score += 10000.0

            if score < best_score:

                best_score = score
                best = candidate.copy()

        if best is None:

            return np.zeros(
                3,
                dtype=float,
            )

        # Soft repulsion activates before the hard boundary.
        repulsion = np.zeros(
            3,
            dtype=float,
        )

        for neighbor in neighbors.values():

            if neighbor.uav_id == uav.uav_id:
                continue

            delta = (
                np.asarray(uav.position)
                - np.asarray(neighbor.position)
            )

            distance = _norm(delta)

            if (
                0.0 < distance < self.soft_radius_m
            ):

                away = delta / distance

                factor = (
                    (
                        self.soft_radius_m
                        - distance
                    )
                    / self.soft_radius_m
                )

                # Relays absorb less repulsion.
                responsibility = (
                    0.15
                    if uav.role == UAVRole.RELAY
                    else 0.75
                )

                repulsion += (
                    away
                    * factor
                    * self.max_speed_mps
                    * 0.35
                    * responsibility
                )

        result = best + repulsion

        speed = _norm(result)

        if speed > self.max_speed_mps:

            result *= (
                self.max_speed_mps
                / speed
            )

        return result

    # ------------------------------------------------------------------
    # High-frequency swarm physics
    # ------------------------------------------------------------------

    def advance(
        self,
        state,
        dt_s: float,
    ) -> MotionDiagnostics:

        if dt_s <= 0.0:
            raise ValueError(
                "dt_s must be positive"
            )

        active = {
            uid: uav
            for uid, uav
            in state.uavs.items()
            if not getattr(
                uav,
                "failure_state",
                False,
            )
            and uav.role not in {
                UAVRole.FAILED,
                UAVRole.LANDING,
            }
        }

        snapshot_positions = {
            uid: np.asarray(
                uav.position,
                dtype=float,
            ).copy()
            for uid, uav in active.items()
        }

        # Simultaneous decisions from the same snapshot.
        planned = {}

        for uid, uav in active.items():

            neighbors = {
                other_uid: other
                for other_uid, other
                in active.items()
                if other_uid != uid
            }

            preferred = self.preferred_velocity(
                uav
            )

            planned[uid] = self.safe_velocity(
                uav,
                preferred,
                neighbors,
            )

        # Apply acceleration/position integration.
        for uid, uav in active.items():

            previous = np.asarray(
                uav.position,
                dtype=float,
            ).copy()

            desired = planned[uid]

            velocity_delta = (
                desired
                - np.asarray(
                    uav.velocity,
                    dtype=float,
                )
            )

            max_delta = (
                self.max_acceleration_mps2
                * dt_s
            )

            delta_norm = _norm(
                velocity_delta
            )

            if delta_norm > max_delta:

                velocity_delta *= (
                    max_delta
                    / delta_norm
                )

            new_velocity = (
                np.asarray(
                    uav.velocity,
                    dtype=float,
                )
                + velocity_delta
            )

            speed = _norm(
                new_velocity
            )

            if speed > self.max_speed_mps:

                new_velocity *= (
                    self.max_speed_mps
                    / speed
                )

            new_position = (
                np.asarray(
                    uav.position,
                    dtype=float,
                )
                + new_velocity * dt_s
            )

            # Geofence.
            clipped = new_position.copy()

            clipped[0] = np.clip(
                clipped[0],
                0.0,
                self.width_m,
            )

            clipped[1] = np.clip(
                clipped[1],
                0.0,
                self.height_m,
            )

            clipped[2] = np.clip(
                clipped[2],
                0.0,
                self.max_altitude_m,
            )

            if not np.allclose(
                clipped,
                new_position,
            ):

                self.diagnostics.geofence_interventions += 1

                for axis, lower, upper in (
                    (0, 0.0, self.width_m),
                    (1, 0.0, self.height_m),
                    (2, 0.0, self.max_altitude_m),
                ):

                    if (
                        clipped[axis]
                        <= lower + EPS
                        and new_velocity[axis] < 0.0
                    ):

                        new_velocity[axis] = 0.0

                    if (
                        clipped[axis]
                        >= upper - EPS
                        and new_velocity[axis] > 0.0
                    ):

                        new_velocity[axis] = 0.0

            uav.position[:] = clipped

            uav.velocity[:] = new_velocity

            uav.acceleration[:] = (
                velocity_delta / dt_s
            )

            uav.distance_travelled_m += _norm(
                uav.position - previous
            )

        # Hard separation pass.
        #
        # This is a final safety layer, not the primary avoidance
        # mechanism. Predictive avoidance should normally prevent it.
        ids = list(active)

        for i in range(len(ids)):

            for j in range(i + 1, len(ids)):

                a = active[ids[i]]
                b = active[ids[j]]

                delta = (
                    np.asarray(a.position)
                    - np.asarray(b.position)
                )

                distance = _norm(delta)

                if distance < self.min_separation_m:

                    if distance <= EPS:

                        axis = (
                            1.0
                            if (
                                a.uav_id
                                < b.uav_id
                            )
                            else -1.0
                        )

                        direction = np.array(
                            [
                                axis,
                                0.0,
                                0.0,
                            ],
                            dtype=float,
                        )

                    else:

                        direction = (
                            delta / distance
                        )

                    correction = (
                        self.min_separation_m
                        + 0.10
                        - distance
                    )

                    # Survey UAVs absorb more of the correction;
                    # relays stay closer to their current station.
                    a_share = (
                        0.75
                        if a.role != UAVRole.RELAY
                        else 0.25
                    )

                    b_share = 1.0 - a_share

                    a.position[:] += (
                        direction
                        * correction
                        * a_share
                    )

                    b.position[:] -= (
                        direction
                        * correction
                        * b_share
                    )

                    self.diagnostics.collision_interventions += 1

                    # Remove closing velocity.
                    relative_velocity = (
                        np.asarray(a.velocity)
                        - np.asarray(b.velocity)
                    )

                    closing = float(
                        np.dot(
                            relative_velocity,
                            direction,
                        )
                    )

                    if closing < 0.0:

                        delta_velocity = (
                            -closing
                        )

                        a.velocity[:] -= (
                            direction
                            * delta_velocity
                            * 0.5
                        )

                        b.velocity[:] += (
                            direction
                            * delta_velocity
                            * 0.5
                        )

        # Final diagnostic pass.
        minimum = float("inf")
        maximum_speed = 0.0

        for uid, uav in active.items():

            speed = _norm(
                np.asarray(
                    uav.velocity,
                    dtype=float,
                )
            )

            maximum_speed = max(
                maximum_speed,
                speed,
            )

            for other_uid, other in active.items():

                if other_uid <= uid:
                    continue

                distance = _norm(
                    np.asarray(
                        uav.position,
                        dtype=float,
                    )
                    - np.asarray(
                        other.position,
                        dtype=float,
                    )
                )

                minimum = min(
                    minimum,
                    distance,
                )

        if minimum == float("inf"):
            minimum = 0.0

        self.diagnostics.min_separation_m = min(
            self.diagnostics.min_separation_m,
            minimum
            if minimum > 0.0
            else self.diagnostics.min_separation_m,
        )

        self.diagnostics.max_speed_mps = max(
            self.diagnostics.max_speed_mps,
            maximum_speed,
        )

        return self.diagnostics

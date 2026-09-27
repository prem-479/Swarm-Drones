from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DynamicsLimits:
    max_speed_mps: float = 5.0
    max_acceleration_mps2: float = 1.5
    max_deceleration_mps2: float = 2.0
    max_vertical_speed_mps: float = 2.5
    max_yaw_rate_rad_s: float = 1.8

    # Aerodynamic response.
    drag_coefficient: float = 0.10
    velocity_response_s: float = 0.65

    # Predictive collision model.
    min_separation_m: float = 20.0
    collision_horizon_s: float = 3.0

    # Small buffer above the hard safety constraint.
    safety_buffer_m: float = 4.0


@dataclass(frozen=True)
class DynamicStep:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    minimum_predicted_separation_m: float
    collision_avoided: bool


class PredictiveVelocityAvoider:
    """
    Velocity-space local planner.

    Candidate velocities are evaluated against predicted
    relative trajectories.  Relay UAVs have higher maneuver
    protection, while survey/recovery UAVs accept more of the
    avoidance responsibility.
    """

    def __init__(
        self,
        limits: DynamicsLimits,
    ) -> None:
        self.limits = limits

    def _responsibility(self, role: str) -> float:
        role = str(role).upper()

        if "RELAY" in role:
            return 0.25

        if "RECOVERY" in role:
            return 0.75

        if "SURVEY" in role:
            return 1.0

        if "TRANSIT" in role:
            return 0.9

        return 0.8

    def _candidate_velocities(
        self,
        preferred: np.ndarray,
        current: np.ndarray,
    ) -> list[np.ndarray]:

        preferred = np.asarray(
            preferred,
            dtype=float,
        ).copy()

        current = np.asarray(
            current,
            dtype=float,
        )

        horizontal_preferred = preferred[:2]

        speed = float(
            np.linalg.norm(
                horizontal_preferred
            )
        )

        if speed > self.limits.max_speed_mps:
            horizontal_preferred *= (
                self.limits.max_speed_mps
                / speed
            )

        if speed < 1e-9:
            base_angle = float(
                np.arctan2(
                    current[1],
                    current[0],
                )
            )
        else:
            base_angle = float(
                np.arctan2(
                    horizontal_preferred[1],
                    horizontal_preferred[0],
                )
            )

        candidates: list[np.ndarray] = []

        speed_scales = (
            0.0,
            0.35,
            0.60,
            0.80,
            1.00,
        )

        angle_offsets_deg = (
            -75.0,
            -50.0,
            -30.0,
            -15.0,
            0.0,
            15.0,
            30.0,
            50.0,
            75.0,
        )

        for scale in speed_scales:
            for offset_deg in angle_offsets_deg:

                angle = (
                    base_angle
                    + np.deg2rad(offset_deg)
                )

                v = np.array(
                    [
                        np.cos(angle),
                        np.sin(angle),
                        preferred[2],
                    ],
                    dtype=float,
                )

                v[:2] *= (
                    min(
                        self.limits.max_speed_mps,
                        speed
                        if speed > 1e-9
                        else self.limits.max_speed_mps,
                    )
                    * scale
                )

                v[2] = np.clip(
                    v[2],
                    -self.limits.max_vertical_speed_mps,
                    self.limits.max_vertical_speed_mps,
                )

                candidates.append(v)

        # Always include preferred velocity.
        candidates.append(
            np.asarray(preferred, dtype=float)
        )

        # Full stop is important during dense encounters.
        candidates.append(
            np.zeros(3, dtype=float)
        )

        return candidates

    def _minimum_predicted_distance(
        self,
        own_position: np.ndarray,
        candidate_velocity: np.ndarray,
        neighbors: Mapping[int, tuple[np.ndarray, np.ndarray]],
    ) -> float:

        minimum_distance = float("inf")

        for neighbor_position, neighbor_velocity in neighbors.values():

            relative_position = (
                np.asarray(neighbor_position, dtype=float)
                - np.asarray(own_position, dtype=float)
            )

            relative_velocity = (
                np.asarray(neighbor_velocity, dtype=float)
                - np.asarray(candidate_velocity, dtype=float)
            )

            rv2 = float(
                np.dot(
                    relative_velocity,
                    relative_velocity,
                )
            )

            if rv2 < 1e-12:
                t_star = 0.0
            else:
                t_star = -float(
                    np.dot(
                        relative_position,
                        relative_velocity,
                    )
                ) / rv2

                t_star = float(
                    np.clip(
                        t_star,
                        0.0,
                        self.limits.collision_horizon_s,
                    )
                )

            predicted = (
                relative_position
                + relative_velocity * t_star
            )

            distance = float(
                np.linalg.norm(predicted)
            )

            minimum_distance = min(
                minimum_distance,
                distance,
            )

        return minimum_distance

    def choose(
        self,
        own_position: np.ndarray,
        current_velocity: np.ndarray,
        preferred_velocity: np.ndarray,
        neighbors: Mapping[int, tuple[np.ndarray, np.ndarray]],
        role: str,
    ) -> tuple[np.ndarray, float, bool]:

        preferred_velocity = np.asarray(
            preferred_velocity,
            dtype=float,
        )

        responsibility = self._responsibility(
            role
        )

        safety_limit = (
            self.limits.min_separation_m
            + self.limits.safety_buffer_m
        )

        best_velocity = None
        best_score = float("inf")
        best_distance = 0.0

        for candidate in self._candidate_velocities(
            preferred_velocity,
            current_velocity,
        ):

            minimum_distance = (
                self._minimum_predicted_distance(
                    own_position,
                    candidate,
                    neighbors,
                )
            )

            unsafe = (
                minimum_distance
                < safety_limit
            )

            # Hard rejection for predicted collisions.
            if unsafe and minimum_distance > 0.0:
                continue

            deviation = float(
                np.linalg.norm(
                    candidate
                    - preferred_velocity
                )
            )

            speed_loss = max(
                0.0,
                self.limits.max_speed_mps
                - float(
                    np.linalg.norm(candidate)
                ),
            )

            maneuver_cost = (
                deviation
                * (0.5 + responsibility)
            )

            score = (
                maneuver_cost
                + 0.15 * speed_loss
            )

            if score < best_score:
                best_score = score
                best_velocity = candidate
                best_distance = minimum_distance

        if best_velocity is not None:
            return (
                best_velocity,
                best_distance,
                not np.allclose(
                    best_velocity,
                    preferred_velocity,
                ),
            )

        # Emergency fallback.
        escape = np.zeros(3, dtype=float)

        own_position = np.asarray(
            own_position,
            dtype=float,
        )

        for neighbor_position, _ in neighbors.values():

            delta = (
                own_position
                - np.asarray(
                    neighbor_position,
                    dtype=float,
                )
            )

            distance = float(
                np.linalg.norm(delta)
            )

            if (
                1e-9
                < distance
                < self.limits.min_separation_m
                * 2.0
            ):

                escape += (
                    delta / distance
                    * (
                        self.limits.min_separation_m
                        * 2.0
                        - distance
                    )
                )

        escape_norm = float(
            np.linalg.norm(escape)
        )

        if escape_norm > 1e-9:

            escape = (
                escape / escape_norm
                * self.limits.max_speed_mps
            )

        else:

            escape = -np.asarray(
                current_velocity,
                dtype=float,
            )

        return (
            escape,
            0.0,
            True,
        )


class HighFidelityFlightDynamics:
    """
    Continuous 2.5D UAV flight model.

    Order:

        preferred velocity
            ↓
        predictive avoidance
            ↓
        acceleration response
            ↓
        aerodynamic drag
            ↓
        wind disturbance
            ↓
        speed limit
            ↓
        turn-rate limit
            ↓
        altitude dynamics
            ↓
        position integration
            ↓
        geofence
    """

    def __init__(
        self,
        limits: DynamicsLimits | None = None,
    ) -> None:

        self.limits = (
            limits
            or DynamicsLimits()
        )

        self.avoider = (
            PredictiveVelocityAvoider(
                self.limits
            )
        )

    @staticmethod
    def _limit_norm(
        vector: np.ndarray,
        maximum: float,
    ) -> np.ndarray:

        vector = np.asarray(
            vector,
            dtype=float,
        ).copy()

        norm = float(
            np.linalg.norm(vector)
        )

        if norm > maximum and norm > 1e-12:

            vector *= (
                maximum / norm
            )

        return vector

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        preferred_velocity: np.ndarray,
        role: str,
        dt_s: float,
        neighbors: Mapping[
            int,
            tuple[np.ndarray, np.ndarray],
        ] | None = None,
        lower_bound: np.ndarray | None = None,
        upper_bound: np.ndarray | None = None,
        wind_velocity: np.ndarray | None = None,
    ) -> DynamicStep:

        if dt_s <= 0.0:
            raise ValueError(
                "dt_s must be positive"
            )

        position = np.asarray(
            position,
            dtype=float,
        ).copy()

        velocity = np.asarray(
            velocity,
            dtype=float,
        ).copy()

        preferred_velocity = np.asarray(
            preferred_velocity,
            dtype=float,
        ).copy()

        neighbors = neighbors or {}

        wind_velocity = (
            np.zeros(3, dtype=float)
            if wind_velocity is None
            else np.asarray(
                wind_velocity,
                dtype=float,
            )
        )

        # -----------------------------------------------------
        # Predictive velocity obstacle layer.
        # -----------------------------------------------------

        safe_velocity, predicted_min_distance, avoided = (
            self.avoider.choose(
                own_position=position,
                current_velocity=velocity,
                preferred_velocity=preferred_velocity,
                neighbors=neighbors,
                role=role,
            )
        )

        # -----------------------------------------------------
        # Velocity tracking / acceleration dynamics.
        # -----------------------------------------------------

        velocity_error = (
            safe_velocity
            - velocity
        )

        desired_acceleration = (
            velocity_error
            / self.limits.velocity_response_s
        )

        # Drag acts against air-relative velocity.
        relative_air_velocity = (
            velocity
            - wind_velocity
        )

        speed = float(
            np.linalg.norm(
                relative_air_velocity
            )
        )

        drag = (
            -self.limits.drag_coefficient
            * speed
            * relative_air_velocity
        )

        desired_acceleration += drag

        # Role-dependent maneuverability.
        role_upper = str(role).upper()

        if "RELAY" in role_upper:

            acceleration_limit = (
                self.limits.max_acceleration_mps2
                * 0.70
            )

        elif "SURVEY" in role_upper:

            acceleration_limit = (
                self.limits.max_acceleration_mps2
            )

        else:

            acceleration_limit = (
                self.limits.max_acceleration_mps2
                * 0.90
            )

        acceleration = self._limit_norm(
            desired_acceleration,
            acceleration_limit,
        )

        # -----------------------------------------------------
        # Integrate velocity.
        # -----------------------------------------------------

        new_velocity = (
            velocity
            + acceleration * dt_s
            + wind_velocity * 0.08 * dt_s
        )

        # Separate acceleration / braking limits.
        current_speed = float(
            np.linalg.norm(velocity)
        )

        new_speed = float(
            np.linalg.norm(new_velocity)
        )

        if new_speed > self.limits.max_speed_mps:

            new_velocity = self._limit_norm(
                new_velocity,
                self.limits.max_speed_mps,
            )

        elif (
            current_speed > 0.0
            and new_speed < current_speed
        ):

            maximum_speed_drop = (
                self.limits.max_deceleration_mps2
                * dt_s
            )

            minimum_allowed_speed = max(
                0.0,
                current_speed
                - maximum_speed_drop,
            )

            if new_speed < minimum_allowed_speed:

                if new_speed > 1e-12:

                    new_velocity *= (
                        minimum_allowed_speed
                        / new_speed
                    )

                else:

                    direction = (
                        velocity
                        / current_speed
                    )

                    new_velocity = (
                        direction
                        * minimum_allowed_speed
                    )

        # -----------------------------------------------------
        # Horizontal turn-rate limitation.
        # -----------------------------------------------------

        old_horizontal = velocity[:2]
        new_horizontal = new_velocity[:2]

        old_speed = float(
            np.linalg.norm(old_horizontal)
        )

        new_horizontal_speed = float(
            np.linalg.norm(new_horizontal)
        )

        if (
            old_speed > 0.1
            and new_horizontal_speed > 0.1
        ):

            old_angle = float(
                np.arctan2(
                    old_horizontal[1],
                    old_horizontal[0],
                )
            )

            new_angle = float(
                np.arctan2(
                    new_horizontal[1],
                    new_horizontal[0],
                )
            )

            angle_delta = (
                new_angle
                - old_angle
                + np.pi
            ) % (
                2.0 * np.pi
            ) - np.pi

            max_turn = (
                self.limits.max_yaw_rate_rad_s
                * dt_s
            )

            angle_delta = float(
                np.clip(
                    angle_delta,
                    -max_turn,
                    max_turn,
                )
            )

            limited_angle = (
                old_angle
                + angle_delta
            )

            new_velocity[0] = (
                np.cos(limited_angle)
                * new_horizontal_speed
            )

            new_velocity[1] = (
                np.sin(limited_angle)
                * new_horizontal_speed
            )

        # Vertical speed limit.
        new_velocity[2] = float(
            np.clip(
                new_velocity[2],
                -self.limits.max_vertical_speed_mps,
                self.limits.max_vertical_speed_mps,
            )
        )

        # -----------------------------------------------------
        # Integrate position.
        # -----------------------------------------------------

        new_position = (
            position
            + 0.5
            * (
                velocity
                + new_velocity
            )
            * dt_s
        )

        # -----------------------------------------------------
        # Geofence.
        # -----------------------------------------------------

        if (
            lower_bound is not None
            and upper_bound is not None
        ):

            new_position = np.clip(
                new_position,
                np.asarray(
                    lower_bound,
                    dtype=float,
                ),
                np.asarray(
                    upper_bound,
                    dtype=float,
                ),
            )

        # If clipping occurred, suppress the outward velocity.
        if lower_bound is not None:

            lower = np.asarray(
                lower_bound,
                dtype=float,
            )

            for axis in range(3):

                if (
                    new_position[axis]
                    <= lower[axis]
                    and new_velocity[axis] < 0
                ):

                    new_velocity[axis] = 0.0

        if upper_bound is not None:

            upper = np.asarray(
                upper_bound,
                dtype=float,
            )

            for axis in range(3):

                if (
                    new_position[axis]
                    >= upper[axis]
                    and new_velocity[axis] > 0
                ):

                    new_velocity[axis] = 0.0

        # Recompute actual acceleration after integration.
        actual_acceleration = (
            new_velocity
            - velocity
        ) / dt_s

        return DynamicStep(
            position=new_position,
            velocity=new_velocity,
            acceleration=actual_acceleration,
            minimum_predicted_separation_m=predicted_min_distance,
            collision_avoided=avoided,
        )

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import pygame


# ============================================================
# UAV-X DYNAMIC SWARM SIMULATOR
# ============================================================

SEED = 42
random.seed(SEED)

SCREEN_W = 1500
SCREEN_H = 900
SIDEBAR_W = 340
MAP_W = SCREEN_W - SIDEBAR_W
MAP_H = SCREEN_H

FPS = 60

WORLD_W = 1000.0
WORLD_H = 1000.0

UAV_COUNT = 10
POI_COUNT = 10

MAX_SPEED = 5.0
MAX_ACCEL = 1.5
MAX_DECEL = 2.0

MAX_ALTITUDE = 100.0
CRUISE_ALTITUDE = 35.0

COMM_RANGE = 100.0
MIN_SEPARATION = 20.0

ENDURANCE = 1200.0
MISSION_DURATION = 2700.0

GCS = pygame.Vector2(50.0, 500.0)

SIM_SPEED = 12.0

BACKGROUND = (8, 12, 18)
MAP_BG = (13, 20, 29)
GRID = (28, 40, 54)

WHITE = (230, 236, 243)
MUTED = (135, 148, 164)

GCS_COLOR = (255, 190, 55)
RELAY_COLOR = (40, 210, 255)
SURVEY_COLOR = (70, 235, 110)
RESERVE_COLOR = (155, 165, 180)
FAILED_COLOR = (255, 70, 80)
POI_COLOR = (255, 75, 190)
DONE_COLOR = (90, 245, 130)
LINK_COLOR = (45, 170, 225)
WIND_COLOR = (150, 120, 255)
TRAIL_COLOR = (55, 75, 95)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def norm(v: pygame.Vector2) -> pygame.Vector2:
    length = v.length()

    if length < 1e-9:
        return pygame.Vector2()

    return v / length


def rotate_toward(
    current: pygame.Vector2,
    desired: pygame.Vector2,
    max_angle: float,
) -> pygame.Vector2:

    if current.length() < 1e-9:
        return desired.normalize() if desired.length() else pygame.Vector2()

    if desired.length() < 1e-9:
        return current.normalize()

    a = math.atan2(current.y, current.x)
    b = math.atan2(desired.y, desired.x)

    delta = (b - a + math.pi) % (2 * math.pi) - math.pi

    delta = clamp(delta, -max_angle, max_angle)

    angle = a + delta

    return pygame.Vector2(
        math.cos(angle),
        math.sin(angle),
    )


@dataclass
class PoI:

    poi_id: int
    position: pygame.Vector2

    priority: int

    service_time: float = 10.0

    assigned_uav: int | None = None

    progress: float = 0.0

    completed: bool = False

    expired: bool = False


@dataclass
class UAV:

    uav_id: int

    position: pygame.Vector2

    velocity: pygame.Vector2 = field(
        default_factory=pygame.Vector2
    )

    acceleration: pygame.Vector2 = field(
        default_factory=pygame.Vector2
    )

    heading: float = 0.0

    altitude: float = CRUISE_ALTITUDE

    vertical_speed: float = 0.0

    battery: float = ENDURANCE

    role: str = "SURVEY"

    target: pygame.Vector2 | None = None

    assigned_poi: int | None = None

    active: bool = True

    failed: bool = False

    rth: bool = False

    trail: List[pygame.Vector2] = field(
        default_factory=list
    )

    role_timer: float = 0.0

    def speed(self) -> float:
        return self.velocity.length()


class DynamicSimulation:

    def __init__(self):

        self.time = 0.0

        self.running = True
        self.paused = False

        self.sim_speed = SIM_SPEED

        self.show_links = True
        self.show_trails = True
        self.show_velocity = True
        self.show_wind = True

        self.wind = pygame.Vector2()

        self.wind_target = pygame.Vector2()

        self.event_log: List[str] = []

        self.uavs: List[UAV] = []

        self.pois: List[PoI] = []

        self.reset()

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        self.time = 0.0

        self.paused = False

        self.wind = pygame.Vector2()

        self.wind_target = pygame.Vector2()

        self.event_log.clear()

        self.uavs.clear()

        self.pois.clear()

        # Initial deployment along a communication backbone.
        for i in range(UAV_COUNT):

            x = 85.0 + i * 82.0

            y = (
                500.0
                + 25.0 * math.sin(i * 0.8)
            )

            if i == 0:
                role = "RELAY"
            else:
                role = "SURVEY"

            uav = UAV(
                uav_id=i,
                position=pygame.Vector2(x, y),
                role=role,
                altitude=CRUISE_ALTITUDE,
            )

            angle = (
                math.atan2(
                    500.0 - y,
                    500.0 - x,
                )
            )

            uav.heading = angle

            self.uavs.append(uav)

        # Deterministic PoI distribution.
        poi_positions = [
            (160, 350),
            (230, 680),
            (315, 420),
            (390, 700),
            (470, 300),
            (560, 650),
            (650, 390),
            (735, 720),
            (830, 360),
            (900, 610),
        ]

        priorities = [
            3, 2, 1, 3, 2,
            1, 3, 2, 1, 3
        ]

        for i in range(POI_COUNT):

            self.pois.append(
                PoI(
                    poi_id=i,
                    position=pygame.Vector2(
                        poi_positions[i]
                    ),
                    priority=priorities[i],
                    service_time=10.0,
                )
            )

        self.log(
            "Scenario initialized"
        )

        self.log(
            "10 UAVs deployed"
        )

        self.log(
            "10 disaster PoIs active"
        )

        self.allocate_tasks()

    # ============================================================
    # EVENT LOG
    # ============================================================

    def log(self, message):

        line = (
            f"{self.time:06.1f}s  "
            f"{message}"
        )

        self.event_log.append(line)

        self.event_log = (
            self.event_log[-9:]
        )

    # ============================================================
    # DISTANCE
    # ============================================================

    @staticmethod
    def distance(
        a: pygame.Vector2,
        b: pygame.Vector2,
    ) -> float:

        return a.distance_to(b)

    # ============================================================
    # COMMUNICATION
    # ============================================================

    def active_uavs(self):

        return [
            uav
            for uav in self.uavs
            if uav.active and not uav.failed
        ]

    def connected_to_gcs(
        self,
        uav: UAV,
    ) -> bool:

        visited = {uav.uav_id}

        frontier = [uav]

        while frontier:

            current = frontier.pop()

            if (
                self.distance(
                    current.position,
                    GCS,
                )
                <= COMM_RANGE
            ):
                return True

            for other in self.active_uavs():

                if other.uav_id in visited:
                    continue

                if (
                    self.distance(
                        current.position,
                        other.position,
                    )
                    <= COMM_RANGE
                ):

                    visited.add(
                        other.uav_id
                    )

                    frontier.append(other)

        return False

    def link_edges(self):

        edges = []

        active = self.active_uavs()

        for i, a in enumerate(active):

            if (
                self.distance(
                    a.position,
                    GCS,
                )
                <= COMM_RANGE
            ):

                edges.append(
                    (
                        "GCS",
                        a.uav_id,
                    )
                )

            for b in active[i + 1:]:

                if (
                    self.distance(
                        a.position,
                        b.position,
                    )
                    <= COMM_RANGE
                ):

                    edges.append(
                        (
                            a.uav_id,
                            b.uav_id,
                        )
                    )

        return edges

    def network_connected_count(self):

        return sum(
            self.connected_to_gcs(uav)
            for uav in self.active_uavs()
        )

    # ============================================================
    # TASK ALLOCATION
    # ============================================================

    def allocate_tasks(self):

        available = [
            uav
            for uav in self.active_uavs()
            if (
                uav.assigned_poi is None
                and not uav.rth
                and uav.role != "RELAY"
            )
        ]

        pending = [
            poi
            for poi in self.pois
            if (
                not poi.completed
                and not poi.expired
                and poi.assigned_uav is None
            )
        ]

        pending.sort(
            key=lambda p: (
                -p.priority,
                p.poi_id,
            )
        )

        for poi in pending:

            if not available:
                break

            best = min(
                available,
                key=lambda u: self.distance(
                    u.position,
                    poi.position,
                ),
            )

            # Do not consume the only relay in the backbone.
            if (
                best.role == "RELAY"
            ):
                continue

            best.assigned_poi = (
                poi.poi_id
            )

            poi.assigned_uav = (
                best.uav_id
            )

            best.target = (
                poi.position.copy()
            )

            available.remove(best)

            self.log(
                f"UAV-{best.uav_id} -> PoI-{poi.poi_id}"
            )

    # ============================================================
    # WIND
    # ============================================================

    def update_wind(self, dt):

        if (
            int(self.time) % 20 == 0
            and random.random() < 0.15
        ):

            angle = random.uniform(
                0.0,
                2.0 * math.pi,
            )

            magnitude = random.uniform(
                0.2,
                0.8,
            )

            self.wind_target = (
                pygame.Vector2(
                    math.cos(angle),
                    math.sin(angle),
                )
                * magnitude
            )

        self.wind = self.wind.lerp(
            self.wind_target,
            clamp(
                dt * 0.5,
                0.0,
                1.0,
            ),
        )

    # ============================================================
    # FAILURE / RECOVERY
    # ============================================================

    def handle_scenario_events(self):

        # UAV-6 fails at 500 s.
        if (
            self.time >= 500.0
            and not getattr(
                self,
                "failure_6_triggered",
                False,
            )
        ):

            self.failure_6_triggered = True

            uav = self.uavs[6]

            uav.failed = True
            uav.active = False
            uav.velocity = pygame.Vector2()

            self.log(
                "UAV-6 FAILURE DETECTED"
            )

            self.reconfigure_after_failure()

        # Second disturbance.
        if (
            self.time >= 1050.0
            and not getattr(
                self,
                "failure_3_triggered",
                False,
            )
        ):

            self.failure_3_triggered = True

            uav = self.uavs[3]

            uav.failed = True
            uav.active = False
            uav.velocity = pygame.Vector2()

            self.log(
                "UAV-3 FAILURE DETECTED"
            )

            self.reconfigure_after_failure()

    def reconfigure_after_failure(self):

        candidates = [
            uav
            for uav in self.active_uavs()
            if (
                uav.role == "SURVEY"
                and not uav.rth
            )
        ]

        if not candidates:
            self.log(
                "No immediate relay replacement"
            )
            return

        replacement = min(
            candidates,
            key=lambda u: u.position.x,
        )

        replacement.role = "RELAY"

        replacement.role_timer = 0.0

        replacement.assigned_poi = None

        replacement.target = pygame.Vector2(
            replacement.position
        )

        self.log(
            f"UAV-{replacement.uav_id} "
            f"promoted to RELAY"
        )

        self.allocate_tasks()

    # ============================================================
    # BATTERY
    # ============================================================

    def update_battery(
        self,
        uav: UAV,
        dt: float,
    ):

        # Dynamic energy approximation:
        # hovering + movement + acceleration.
        movement_factor = (
            uav.speed() / MAX_SPEED
        )

        acceleration_factor = (
            uav.acceleration.length()
            / MAX_ACCEL
        )

        consumption = (
            0.25
            + 0.20 * movement_factor
            + 0.10 * acceleration_factor
        )

        uav.battery -= (
            consumption * dt
        )

        if uav.battery < 180.0 and not uav.rth:

            uav.rth = True

            uav.role = "RTH"

            uav.assigned_poi = None

            uav.target = (
                GCS.copy()
            )

            self.log(
                f"UAV-{uav.uav_id} "
                f"LOW BATTERY -> RTH"
            )

    # ============================================================
    # COLLISION AVOIDANCE
    # ============================================================

    def collision_avoidance(
        self,
        uav: UAV,
    ) -> pygame.Vector2:

        correction = pygame.Vector2()

        for other in self.active_uavs():

            if other.uav_id == uav.uav_id:
                continue

            delta = (
                uav.position -
                other.position
            )

            distance = delta.length()

            if (
                distance > 0
                and distance < MIN_SEPARATION * 1.8
            ):

                strength = (
                    MIN_SEPARATION * 1.8
                    - distance
                ) / (
                    MIN_SEPARATION * 1.8
                )

                correction += (
                    delta.normalize()
                    * strength
                    * MAX_SPEED
                )

        return correction

    # ============================================================
    # NAVIGATION
    # ============================================================

    def desired_velocity(
        self,
        uav: UAV,
    ) -> pygame.Vector2:

        if uav.rth:

            target = GCS

        elif uav.target is not None:

            target = uav.target

        else:

            # Survey behavior:
            # move around the assigned region.
            angle = (
                self.time * 0.08
                + uav.uav_id * 0.6
            )

            target = (
                pygame.Vector2(
                    500.0,
                    500.0,
                )
                + pygame.Vector2(
                    math.cos(angle),
                    math.sin(angle),
                )
                * 300.0
            )

        direction = (
            target -
            uav.position
        )

        desired = norm(direction)

        # Add wind compensation.
        desired_velocity = (
            desired * MAX_SPEED
        )

        desired_velocity -= (
            self.wind * 0.6
        )

        desired_velocity += (
            self.collision_avoidance(
                uav
            )
        )

        # Geofence braking.
        margin = 45.0

        if uav.position.x < margin:
            desired_velocity.x += 1.0

        if uav.position.x > WORLD_W - margin:
            desired_velocity.x -= 1.0

        if uav.position.y < margin:
            desired_velocity.y += 1.0

        if uav.position.y > WORLD_H - margin:
            desired_velocity.y -= 1.0

        if desired_velocity.length() > MAX_SPEED:
            desired_velocity.scale_to_length(
                MAX_SPEED
            )

        return desired_velocity

    # ============================================================
    # UAV DYNAMICS
    # ============================================================

    def update_uav(
        self,
        uav: UAV,
        dt: float,
    ):

        if not uav.active:
            return

        desired = self.desired_velocity(
            uav
        )

        velocity_error = (
            desired -
            uav.velocity
        )

        if velocity_error.length() > 1e-6:

            max_change = (
                MAX_ACCEL * dt
            )

            if (
                velocity_error.length()
                > max_change
            ):

                acceleration = (
                    velocity_error.normalize()
                    * MAX_ACCEL
                )

            else:

                acceleration = (
                    velocity_error / dt
                )

        else:

            acceleration = pygame.Vector2()

        # Smooth acceleration.
        uav.acceleration = (
            uav.acceleration.lerp(
                acceleration,
                clamp(
                    dt * 4.0,
                    0.0,
                    1.0,
                ),
            )
        )

        uav.velocity += (
            uav.acceleration * dt
        )

        # Limit speed.
        if uav.velocity.length() > MAX_SPEED:

            uav.velocity.scale_to_length(
                MAX_SPEED
            )

        # Smooth heading / turn dynamics.
        if uav.velocity.length() > 0.1:

            direction = (
                uav.velocity.normalize()
            )

            current_dir = pygame.Vector2(
                math.cos(uav.heading),
                math.sin(uav.heading),
            )

            new_dir = rotate_toward(
                current_dir,
                direction,
                math.radians(
                    120.0 * dt
                ),
            )

            uav.heading = math.atan2(
                new_dir.y,
                new_dir.x,
            )

        uav.position += (
            uav.velocity * dt
        )

        # Keep inside geofence.
        uav.position.x = clamp(
            uav.position.x,
            5.0,
            WORLD_W - 5.0,
        )

        uav.position.y = clamp(
            uav.position.y,
            5.0,
            WORLD_H - 5.0,
        )

        # Altitude dynamics.
        desired_altitude = (
            28.0
            if uav.role == "RELAY"
            else CRUISE_ALTITUDE
        )

        if uav.assigned_poi is not None:

            desired_altitude = 32.0

        altitude_error = (
            desired_altitude
            - uav.altitude
        )

        uav.vertical_speed += (
            clamp(
                altitude_error,
                -8.0,
                8.0,
            )
            * dt
        )

        uav.vertical_speed = clamp(
            uav.vertical_speed,
            -5.0,
            5.0,
        )

        uav.altitude += (
            uav.vertical_speed * dt
        )

        uav.altitude = clamp(
            uav.altitude,
            10.0,
            MAX_ALTITUDE,
        )

        self.update_battery(
            uav,
            dt,
        )

        if self.show_trails:

            uav.trail.append(
                uav.position.copy()
            )

            if len(uav.trail) > 100:
                uav.trail.pop(0)

    # ============================================================
    # TASK PROGRESS
    # ============================================================

    def update_tasks(self, dt):

        for poi in self.pois:

            if (
                poi.completed
                or poi.expired
            ):
                continue

            if (
                poi.assigned_uav is None
            ):
                continue

            uav = self.uavs[
                poi.assigned_uav
            ]

            if (
                not uav.active
                or uav.failed
            ):

                poi.assigned_uav = None

                uav.assigned_poi = None
                uav.target = None

                self.allocate_tasks()

                continue

            distance = self.distance(
                uav.position,
                poi.position,
            )

            if distance < 18.0:

                if not hasattr(
                    poi,
                    "start_service",
                ):

                    poi.start_service = (
                        self.time
                    )

                    self.log(
                        f"UAV-{uav.uav_id} "
                        f"reached PoI-{poi.poi_id}"
                    )

                elapsed = (
                    self.time
                    - poi.start_service
                )

                poi.progress = clamp(
                    elapsed /
                    poi.service_time,
                    0.0,
                    1.0,
                )

                if (
                    poi.progress >= 1.0
                ):

                    poi.completed = True

                    uav.assigned_poi = None
                    uav.target = None

                    poi.assigned_uav = None

                    self.log(
                        f"PoI-{poi.poi_id} "
                        f"COMPLETED"
                    )

                    self.allocate_tasks()

    # ============================================================
    # RELAY MANAGEMENT
    # ============================================================

    def update_relays(self):

        active = self.active_uavs()

        disconnected = [
            uav
            for uav in active
            if not self.connected_to_gcs(uav)
        ]

        if disconnected:

            candidate = min(
                active,
                key=lambda u: self.distance(
                    u.position,
                    GCS,
                ),
            )

            if candidate.role != "RELAY":

                candidate.role = "RELAY"

                candidate.role_timer = 0.0

                candidate.assigned_poi = None

                self.log(
                    f"UAV-{candidate.uav_id} "
                    f"assigned RELAY"
                )

                self.allocate_tasks()

        # Protect long-distance backbone.
        sorted_active = sorted(
            active,
            key=lambda u: u.position.x,
        )

        for i in range(
            len(sorted_active) - 1
        ):

            a = sorted_active[i]
            b = sorted_active[i + 1]

            distance = self.distance(
                a.position,
                b.position,
            )

            if (
                distance > COMM_RANGE * 0.85
                and a.role != "RTH"
                and b.role != "RTH"
            ):

                candidate = b

                if (
                    candidate.role
                    != "RELAY"
                ):

                    candidate.role = "RELAY"

                    candidate.role_timer = 0.0

                    candidate.assigned_poi = None

                    candidate.target = pygame.Vector2(
                        (
                            a.position
                            + b.position
                        )
                        / 2.0
                    )

                    self.log(
                        f"UAV-{candidate.uav_id} "
                        f"moved to RELAY"
                    )

                    self.allocate_tasks()

    # ============================================================
    # RETURN HOME
    # ============================================================

    def handle_rth(self):

        for uav in self.active_uavs():

            if not uav.rth:
                continue

            if (
                self.distance(
                    uav.position,
                    GCS,
                )
                < 25.0
            ):

                uav.active = False

                uav.velocity = (
                    pygame.Vector2()
                )

                self.log(
                    f"UAV-{uav.uav_id} "
                    f"LANDED / RTH"
                )

    # ============================================================
    # SIMULATION STEP
    # ============================================================

    def step(self, dt):

        self.time += dt

        self.update_wind(dt)

        self.handle_scenario_events()

        self.update_relays()

        for uav in self.uavs:
            self.update_uav(
                uav,
                dt,
            )

        self.update_tasks(dt)

        self.handle_rth()

    # ============================================================
    # MAPPING
    # ============================================================

    def screen_position(
        self,
        position: pygame.Vector2,
    ):

        sx = int(
            20
            + (
                position.x
                / WORLD_W
            )
            * (
                MAP_W - 40
            )
        )

        sy = int(
            MAP_H
            - (
                20
                + (
                    position.y
                    / WORLD_H
                )
                * (
                    MAP_H - 40
                )
            )
        )

        return sx, sy

    # ============================================================
    # DRAW MAP
    # ============================================================

    def draw_map(
        self,
        screen,
        font,
        small_font,
    ):

        pygame.draw.rect(
            screen,
            MAP_BG,
            (0, 0, MAP_W, MAP_H),
        )

        # Grid.
        for x in range(
            0,
            1001,
            100,
        ):

            sx, _ = self.screen_position(
                pygame.Vector2(
                    x,
                    0,
                )
            )

            pygame.draw.line(
                screen,
                GRID,
                (sx, 20),
                (sx, MAP_H - 20),
            )

        for y in range(
            0,
            1001,
            100,
        ):

            _, sy = self.screen_position(
                pygame.Vector2(
                    0,
                    y,
                )
            )

            pygame.draw.line(
                screen,
                GRID,
                (20, sy),
                (MAP_W - 20, sy),
            )

        # GCS.
        gx, gy = self.screen_position(
            GCS
        )

        pygame.draw.circle(
            screen,
            GCS_COLOR,
            (gx, gy),
            11,
        )

        pygame.draw.circle(
            screen,
            WHITE,
            (gx, gy),
            15,
            2,
        )

        screen.blit(
            small_font.render(
                "GCS",
                True,
                GCS_COLOR,
            ),
            (gx + 18, gy - 9),
        )

        # Communication links.
        if self.show_links:

            for a, b in self.link_edges():

                if a == "GCS":

                    ua = self.uavs[
                        b
                    ]

                    ax, ay = (
                        self.screen_position(
                            ua.position
                        )
                    )

                    bx, by = (
                        self.screen_position(
                            GCS
                        )
                    )

                else:

                    ua = self.uavs[
                        a
                    ]

                    ub = self.uavs[
                        b
                    ]

                    ax, ay = (
                        self.screen_position(
                            ua.position
                        )
                    )

                    bx, by = (
                        self.screen_position(
                            ub.position
                        )
                    )

                pygame.draw.line(
                    screen,
                    LINK_COLOR,
                    (ax, ay),
                    (bx, by),
                    2,
                )

        # Wind.
        if (
            self.show_wind
            and self.wind.length() > 0.05
        ):

            wx, wy = self.screen_position(
                pygame.Vector2(
                    500,
                    850,
                )
            )

            wind_end = (
                pygame.Vector2(
                    wx,
                    wy,
                )
                + self.wind
                * 90.0
            )

            pygame.draw.line(
                screen,
                WIND_COLOR,
                (wx, wy),
                wind_end,
                3,
            )

        # PoIs.
        for poi in self.pois:

            sx, sy = self.screen_position(
                poi.position
            )

            if poi.completed:
                color = DONE_COLOR

            else:
                color = (
                    POI_COLOR
                )

            pygame.draw.circle(
                screen,
                color,
                (sx, sy),
                8,
            )

            pygame.draw.circle(
                screen,
                color,
                (sx, sy),
                13,
                1,
            )

            label = (
                f"P{poi.poi_id} "
                f"[{poi.priority}]"
            )

            screen.blit(
                small_font.render(
                    label,
                    True,
                    color,
                ),
                (sx + 14, sy - 7),
            )

            if poi.progress > 0:

                pygame.draw.rect(
                    screen,
                    DONE_COLOR,
                    (
                        sx - 12,
                        sy + 16,
                        24 * poi.progress,
                        3,
                    ),
                )

        # UAVs.
        for uav in self.uavs:

            sx, sy = self.screen_position(
                uav.position
            )

            if uav.failed:

                color = FAILED_COLOR

            elif uav.role == "RELAY":

                color = RELAY_COLOR

            elif uav.role == "RTH":

                color = GCS_COLOR

            elif uav.role == "SURVEY":

                color = SURVEY_COLOR

            else:

                color = RESERVE_COLOR

            radius = (
                10
                if uav.role == "RELAY"
                else 8
            )

            pygame.draw.circle(
                screen,
                color,
                (sx, sy),
                radius,
            )

            pygame.draw.circle(
                screen,
                WHITE,
                (sx, sy),
                radius,
                1,
            )

            # Heading arrow.
            heading_vec = pygame.Vector2(
                math.cos(uav.heading),
                -math.sin(uav.heading),
            )

            arrow_end = (
                pygame.Vector2(
                    sx,
                    sy,
                )
                + heading_vec
                * 20
            )

            pygame.draw.line(
                screen,
                color,
                (sx, sy),
                arrow_end,
                3,
            )

            # Velocity vector.
            if (
                self.show_velocity
                and uav.speed() > 0.15
            ):

                vel = (
                    uav.velocity
                    * 15.0
                )

                pygame.draw.line(
                    screen,
                    WHITE,
                    (sx, sy),
                    (
                        sx + int(
                            vel.x
                        ),
                        sy - int(
                            vel.y
                        ),
                    ),
                    1,
                )

            # Altitude ring.
            pygame.draw.circle(
                screen,
                color,
                (sx, sy),
                radius + 5,
                1,
            )

            label = (
                f"UAV-{uav.uav_id} "
                f"{uav.role} "
                f"{uav.speed():.1f}m/s"
            )

            screen.blit(
                small_font.render(
                    label,
                    True,
                    color,
                ),
                (sx + 15, sy - 9),
            )

            if uav.assigned_poi is not None:

                screen.blit(
                    small_font.render(
                        f"→ P{uav.assigned_poi}",
                        True,
                        WHITE,
                    ),
                    (sx + 15, sy + 9),
                )

            # Trail.
            if (
                self.show_trails
                and len(uav.trail) > 1
            ):

                points = [
                    self.screen_position(
                        point
                    )
                    for point in uav.trail
                ]

                pygame.draw.lines(
                    screen,
                    TRAIL_COLOR,
                    False,
                    points,
                    1,
                )

    # ============================================================
    # SIDEBAR
    # ============================================================

    def draw_sidebar(
        self,
        screen,
        font,
        small_font,
    ):

        pygame.draw.rect(
            screen,
            (17, 24, 34),
            (
                MAP_W,
                0,
                SIDEBAR_W,
                SCREEN_H,
            ),
        )

        x = MAP_W + 22
        y = 22

        screen.blit(
            font.render(
                "UAV-X",
                True,
                WHITE,
            ),
            (x, y),
        )

        y += 30

        screen.blit(
            small_font.render(
                "RESILIENT BVLOS SWARM",
                True,
                MUTED,
            ),
            (x, y),
        )

        y += 40

        minutes = int(
            self.time // 60
        )

        seconds = int(
            self.time % 60
        )

        screen.blit(
            small_font.render(
                f"MISSION TIME   "
                f"{minutes:02d}:{seconds:02d}",
                True,
                WHITE,
            ),
            (x, y),
        )

        y += 20

        screen.blit(
            small_font.render(
                f"SIM SPEED      "
                f"{self.sim_speed:.0f}x",
                True,
                RELAY_COLOR,
            ),
            (x, y),
        )

        y += 34

        # Mission.
        screen.blit(
            small_font.render(
                "MISSION",
                True,
                MUTED,
            ),
            (x, y),
        )

        y += 21

        completed = sum(
            poi.completed
            for poi in self.pois
        )

        active = sum(
            (
                poi.assigned_uav
                is not None
                and not poi.completed
            )
            for poi in self.pois
        )

        remaining = (
            POI_COUNT
            - completed
        )

        screen.blit(
            small_font.render(
                f"PoI COMPLETED   {completed}/{POI_COUNT}",
                True,
                DONE_COLOR,
            ),
            (x, y),
        )

        y += 19

        screen.blit(
            small_font.render(
                f"PoI ACTIVE      {active}",
                True,
                WHITE,
            ),
            (x, y),
        )

        y += 19

        screen.blit(
            small_font.render(
                f"PoI REMAINING   {remaining}",
                True,
                POI_COLOR,
            ),
            (x, y),
        )

        y += 32

        # Swarm.
        screen.blit(
            small_font.render(
                "SWARM",
                True,
                MUTED,
            ),
            (x, y),
        )

        y += 21

        counts = {
            "RELAY": 0,
            "SURVEY": 0,
            "RTH": 0,
            "FAILED": 0,
        }

        for uav in self.uavs:

            if uav.failed:
                counts["FAILED"] += 1

            elif uav.role == "RELAY":
                counts["RELAY"] += 1

            elif uav.role == "RTH":
                counts["RTH"] += 1

            else:
                counts["SURVEY"] += 1

        for name in (
            "RELAY",
            "SURVEY",
            "RTH",
            "FAILED",
        ):

            screen.blit(
                small_font.render(
                    f"{name:<8} {counts[name]}",
                    True,
                    role_color(name),
                ),
                (x, y),
            )

            y += 19

        y += 18

        # Network.
        screen.blit(
            small_font.render(
                "NETWORK",
                True,
                MUTED,
            ),
            (x, y),
        )

        y += 21

        edges = len(
            self.link_edges()
        )

        connected = (
            self.network_connected_count()
        )

        screen.blit(
            small_font.render(
                f"LINKS          {edges}",
                True,
                LINK_COLOR,
            ),
            (x, y),
        )

        y += 19

        screen.blit(
            small_font.render(
                f"GCS REACHABLE  "
                f"{connected}/{UAV_COUNT}",
                True,
                DONE_COLOR
                if connected
                == len(
                    self.active_uavs()
                )
                else FAILED_COLOR,
            ),
            (x, y),
        )

        y += 32

        # UAV data.
        screen.blit(
            small_font.render(
                "LIVE UAV STATE",
                True,
                MUTED,
            ),
            (x, y),
        )

        y += 21

        for uav in self.uavs:

            if uav.failed:

                value = "FAILED"

            else:

                value = (
                    f"{uav.speed():.1f}m/s "
                    f"{uav.altitude:.0f}m "
                    f"{uav.battery:.0f}s"
                )

            screen.blit(
                small_font.render(
                    f"UAV-{uav.uav_id:<2} "
                    f"{value}",
                    True,
                    role_color(
                        "FAILED"
                        if uav.failed
                        else uav.role
                    ),
                ),
                (x, y),
            )

            y += 17

        y += 12

        screen.blit(
            small_font.render(
                "EVENTS",
                True,
                MUTED,
            ),
            (x, y),
        )

        y += 20

        for event in self.event_log[-7:]:

            screen.blit(
                pygame.font.SysFont(
                    "DejaVu Sans",
                    11,
                ).render(
                    event,
                    True,
                    WHITE,
                ),
                (x, y),
            )

            y += 16

        y = SCREEN_H - 118

        controls = (
            "SPACE  Pause / Resume",
            "+/-    Simulation speed",
            "L      Communication links",
            "T      Trails",
            "V      Velocity vectors",
            "W      Wind",
            "R      Reset",
            "ESC    Exit",
        )

        for line in controls:

            screen.blit(
                small_font.render(
                    line,
                    True,
                    MUTED,
                ),
                (x, y),
            )

            y += 16

    # ============================================================
    # EVENTS
    # ============================================================

    def handle_input(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                self.running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    self.running = False

                elif event.key == pygame.K_SPACE:

                    self.paused = not self.paused

                elif event.key == pygame.K_r:

                    self.reset()

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                ):

                    self.sim_speed = min(
                        100.0,
                        self.sim_speed * 1.5,
                    )

                elif event.key == pygame.K_MINUS:

                    self.sim_speed = max(
                        1.0,
                        self.sim_speed / 1.5,
                    )

                elif event.key == pygame.K_l:

                    self.show_links = (
                        not self.show_links
                    )

                elif event.key == pygame.K_t:

                    self.show_trails = (
                        not self.show_trails
                    )

                elif event.key == pygame.K_v:

                    self.show_velocity = (
                        not self.show_velocity
                    )

                elif event.key == pygame.K_w:

                    self.show_wind = (
                        not self.show_wind
                    )

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def run(self):

        pygame.init()

        screen = pygame.display.set_mode(
            (
                SCREEN_W,
                SCREEN_H,
            )
        )

        pygame.display.set_caption(
            "UAV-X Dynamic Swarm Simulator"
        )

        clock = pygame.time.Clock()

        font = pygame.font.SysFont(
            "DejaVu Sans",
            24,
            bold=True,
        )

        small_font = pygame.font.SysFont(
            "DejaVu Sans",
            14,
        )

        while self.running:

            self.handle_input()

            if not self.paused:

                real_dt = (
                    clock.get_time()
                    / 1000.0
                )

                real_dt = clamp(
                    real_dt,
                    0.001,
                    0.05,
                )

                sim_dt = (
                    real_dt
                    * self.sim_speed
                )

                # Break large accelerated steps into
                # small dynamic integration steps.
                remaining = sim_dt

                while remaining > 0:

                    step_dt = min(
                        remaining,
                        0.1,
                    )

                    self.step(
                        step_dt
                    )

                    remaining -= (
                        step_dt
                    )

            self.draw_map(
                screen,
                font,
                small_font,
            )

            self.draw_sidebar(
                screen,
                font,
                small_font,
            )

            pygame.display.flip()

            clock.tick(FPS)

        pygame.quit()


def role_color(role: str):

    if role == "RELAY":
        return RELAY_COLOR

    if role == "SURVEY":
        return SURVEY_COLOR

    if role == "RTH":
        return GCS_COLOR

    if role == "FAILED":
        return FAILED_COLOR

    return RESERVE_COLOR


def main():

    simulation = DynamicSimulation()

    simulation.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())

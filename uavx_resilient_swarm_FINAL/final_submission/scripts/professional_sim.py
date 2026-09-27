from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from uavx.core.config import load_config
from uavx.core.models import MissionStatus, TaskStatus, UAVRole
from uavx.simulation import SimulationEngine


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "official_stage1.yaml"

WIDTH = 1600
HEIGHT = 920
SIDEBAR = 360
MAP_WIDTH = WIDTH - SIDEBAR
FPS = 60

SIM_SPEED = 40.0
WORLD_W = 1000.0
WORLD_H = 1000.0

# UI palette
BG = (7, 11, 17)
MAP_BG = (11, 17, 25)
PANEL = (15, 22, 32)
PANEL_2 = (20, 29, 41)
GRID = (31, 44, 59)
TEXT = (228, 235, 243)
MUTED = (128, 143, 160)

GCS = (255, 190, 55)
RELAY = (60, 210, 255)
SURVEY = (75, 235, 120)
RESERVE = (160, 173, 190)
FAILED = (255, 75, 85)
RTH = (255, 180, 70)
POI = (255, 85, 190)
DONE = (90, 245, 130)
LINK = (40, 185, 235)
WARNING = (255, 175, 60)
TRAIL = (55, 78, 100)
WHITE = TEXT


def enum_name(value) -> str:
    if value is None:
        return ""
    return str(value).split(".")[-1].upper()


def as_xyz(obj) -> np.ndarray:
    value = getattr(obj, "position", None)

    if value is None:
        return np.zeros(3, dtype=float)

    return np.asarray(value, dtype=float).copy()


def role_colour(role) -> tuple[int, int, int]:
    name = enum_name(role)

    if name == "RELAY":
        return RELAY

    if name in {"RETURN_HOME", "RTH"}:
        return RTH

    if name in {"FAILED", "LANDING"}:
        return FAILED

    if name == "SURVEY":
        return SURVEY

    return RESERVE


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ProfessionalSimulator:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT),
            pygame.DOUBLEBUF,
        )

        pygame.display.set_caption(
            "UAV-X | Resilient BVLOS Swarm Command Center"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "DejaVu Sans",
            21,
        )

        self.small = pygame.font.SysFont(
            "DejaVu Sans",
            14,
        )

        self.tiny = pygame.font.SysFont(
            "DejaVu Sans",
            11,
        )

        self.header = pygame.font.SysFont(
            "DejaVu Sans",
            27,
            bold=True,
        )

        self.running = True
        self.paused = False

        self.sim_speed = SIM_SPEED

        self.show_links = True
        self.show_trails = True
        self.show_vectors = True
        self.show_labels = True
        self.show_grid = True
        self.follow_uav = None

        self.selected_uav = None

        self.trails: dict[int, deque] = {}

        self.previous_roles: dict[int, str] = {}
        self.previous_tasks: dict[int, str] = {}

        self.events = deque(maxlen=9)

        self.reset()

    # ------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------

    def reset(self):

        config = load_config(
            str(CONFIG)
        )

        self.engine = SimulationEngine.from_config(
            config
        )

        self.engine.start()

        # Professional demonstration startup:
        # create six reproducible random PoIs immediately so that
        # the swarm visibly enters mission execution at t=0.
        #
        # The remaining scheduled PoI events remain dynamic and can
        # bring the active population up to the configured limit.
        initial_poi_count = min(
            6,
            self.engine.config.scenario.poi_count,
        )

        for poi in self.engine.scenario_generator.iter_random_pois(
            initial_poi_count,
            timestamp_s=0.0,
        ):
            self.engine.state.pois[poi.poi_id] = poi

        # Execute the autonomy cycle once before the first rendered
        # frame so assignments/roles/targets exist immediately.
        self.engine.swarm.update(
            self.engine.state,
            0.0,
        )

        self.trails.clear()

        self.previous_roles.clear()

        self.previous_tasks.clear()

        self.events.clear()

        self.selected_uav = None
        self.follow_uav = None

        self.log(
            "Simulation initialized"
        )

    def step_simulation(self, real_dt):

        if self.paused:
            return

        mission_status = self.engine.state.mission.status

        if mission_status != MissionStatus.RUNNING:
            return

        sim_dt = min(
            real_dt * self.sim_speed,
            0.80,
        )

        remaining = sim_dt

        while remaining > 0:

            step = min(
                remaining,
                0.10,
            )

            self.engine.step(step)

            remaining -= step

            if (
                self.engine.state.mission.status
                != MissionStatus.RUNNING
            ):
                break

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    def log(self, text):

        timestamp = float(
            getattr(
                self.engine.state,
                "timestamp_s",
                0.0,
            )
        )

        self.events.append(
            f"{timestamp:07.1f}  {text}"
        )

    def detect_events(self):

        for uid, uav in self.engine.state.uavs.items():

            role = enum_name(
                getattr(
                    uav,
                    "role",
                    None,
                )
            )

            previous = self.previous_roles.get(uid)

            if (
                previous is not None
                and previous != role
            ):

                self.log(
                    f"UAV-{uid} "
                    f"{previous} → {role}"
                )

            self.previous_roles[uid] = role

        for task_id, task in self.engine.state.tasks.items():

            status = enum_name(
                getattr(
                    task,
                    "status",
                    None,
                )
            )

            previous = self.previous_tasks.get(task_id)

            if (
                previous is not None
                and previous != status
                and status in {
                    "COMPLETED",
                    "EXPIRED",
                    "ASSIGNED",
                    "IN_PROGRESS",
                }
            ):

                self.log(
                    f"Task-{task_id} → {status}"
                )

            self.previous_tasks[task_id] = status

    # ------------------------------------------------------------
    # Coordinate system
    # ------------------------------------------------------------

    def world_to_screen(
        self,
        x,
        y,
    ):

        sx = 20 + int(
            (
                x / WORLD_W
            )
            * (MAP_WIDTH - 40)
        )

        sy = (
            HEIGHT
            - 20
            - int(
                (
                    y / WORLD_H
                )
                * (HEIGHT - 40)
            )
        )

        return sx, sy

    # ------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------

    def text(
        self,
        value,
        x,
        y,
        colour=TEXT,
        font=None,
    ):

        if font is None:
            font = self.small

        surface = font.render(
            str(value),
            True,
            colour,
        )

        self.screen.blit(
            surface,
            (x, y),
        )

    def line(
        self,
        p1,
        p2,
        colour,
        width=1,
    ):

        pygame.draw.line(
            self.screen,
            colour,
            p1,
            p2,
            width,
        )

    # ------------------------------------------------------------
    # Map
    # ------------------------------------------------------------

    def draw_map(self):

        pygame.draw.rect(
            self.screen,
            MAP_BG,
            (0, 0, MAP_WIDTH, HEIGHT),
        )

        if self.show_grid:

            for x in range(
                0,
                1001,
                100,
            ):

                sx, _ = self.world_to_screen(
                    x,
                    0,
                )

                self.line(
                    (sx, 20),
                    (sx, HEIGHT - 20),
                    GRID,
                    1,
                )

                self.text(
                    f"{x}",
                    sx - 9,
                    HEIGHT - 19,
                    MUTED,
                    self.tiny,
                )

            for y in range(
                0,
                1001,
                100,
            ):

                _, sy = self.world_to_screen(
                    0,
                    y,
                )

                self.line(
                    (20, sy),
                    (MAP_WIDTH - 20, sy),
                    GRID,
                    1,
                )

                self.text(
                    f"{y}",
                    3,
                    sy - 6,
                    MUTED,
                    self.tiny,
                )

        # Boundary
        pygame.draw.rect(
            self.screen,
            GRID,
            (20, 20, MAP_WIDTH - 40, HEIGHT - 40),
            2,
        )

    # ------------------------------------------------------------
    # Network graph
    # ------------------------------------------------------------

    def draw_network(self):

        if not self.show_links:
            return

        network = self.engine.state.network

        positions = {
            uid: as_xyz(uav)
            for uid, uav
            in self.engine.state.uavs.items()
        }

        gcs_position = np.asarray(
            self.engine.gcs_position,
            dtype=float,
        )

        for key, link in getattr(
            network,
            "links",
            {},
        ).items():

            if not getattr(
                link,
                "availability",
                False,
            ):
                continue

            try:
                source, target = key
            except Exception:
                continue

            if source == "GCS":

                if target not in positions:
                    continue

                a = gcs_position
                b = positions[target]

            elif target == "GCS":

                if source not in positions:
                    continue

                a = positions[source]
                b = gcs_position

            else:

                if (
                    source not in positions
                    or target not in positions
                ):
                    continue

                a = positions[source]
                b = positions[target]

            quality = float(
                getattr(
                    link,
                    "link_quality",
                    1.0,
                )
            )

            colour = (
                LINK
                if quality > 0.70
                else WARNING
                if quality > 0.35
                else FAILED
            )

            a2 = self.world_to_screen(
                a[0],
                a[1],
            )

            b2 = self.world_to_screen(
                b[0],
                b[1],
            )

            self.line(
                a2,
                b2,
                colour,
                2,
            )

    # ------------------------------------------------------------
    # GCS
    # ------------------------------------------------------------

    def draw_gcs(self):

        # GCS is outside the operational area. Project its visual
        # anchor onto the left boundary so the command link remains
        # visible without changing the world geometry.
        gx = max(
            0.0,
            float(self.engine.gcs_position[0]),
        )

        gcs = self.world_to_screen(
            gx,
            self.engine.gcs_position[1],
        )

        pygame.draw.circle(
            self.screen,
            GCS,
            gcs,
            12,
        )

        pygame.draw.circle(
            self.screen,
            TEXT,
            gcs,
            17,
            2,
        )

        self.text(
            "GCS",
            gcs[0] + 21,
            gcs[1] - 9,
            GCS,
            self.small,
        )

    # ------------------------------------------------------------
    # PoIs
    # ------------------------------------------------------------

    def task_for_poi(self, poi_id):

        return self.engine.state.tasks.get(
            poi_id
        )

    def draw_pois(self):

        for poi_id, poi in self.engine.state.pois.items():

            position = as_xyz(poi)

            sx, sy = self.world_to_screen(
                position[0],
                position[1],
            )

            task = self.task_for_poi(
                poi_id
            )

            status = enum_name(
                getattr(
                    task,
                    "status",
                    None,
                )
                if task
                else getattr(
                    poi,
                    "status",
                    None,
                )
            )

            if "COMPLETED" in status:
                colour = DONE
            elif "EXPIRED" in status:
                colour = FAILED
            else:
                colour = POI

            pygame.draw.circle(
                self.screen,
                colour,
                (sx, sy),
                8,
            )

            pygame.draw.circle(
                self.screen,
                colour,
                (sx, sy),
                14,
                1,
            )

            priority = getattr(
                poi,
                "priority",
                1,
            )

            label = (
                f"P{poi_id}"
                f"  [{priority}]"
            )

            self.text(
                label,
                sx + 16,
                sy - 8,
                colour,
                self.tiny,
            )

    # ------------------------------------------------------------
    # UAVs
    # ------------------------------------------------------------

    def draw_trail(
        self,
        uid,
    ):

        if not self.show_trails:
            return

        trail = self.trails.get(uid)

        if not trail or len(trail) < 2:
            return

        points = [
            self.world_to_screen(
                p[0],
                p[1],
            )
            for p in trail
        ]

        self.line(
            points[0],
            points[-1],
            TRAIL,
            1,
        )

        pygame.draw.lines(
            self.screen,
            TRAIL,
            False,
            points,
            1,
        )

    def update_trails(self):

        for uid, uav in self.engine.state.uavs.items():

            pos = as_xyz(uav)

            trail = self.trails.setdefault(
                uid,
                deque(maxlen=90),
            )

            trail.append(
                (
                    float(pos[0]),
                    float(pos[1]),
                )
            )

    def draw_uav(
        self,
        uid,
        uav,
    ):

        pos = as_xyz(uav)

        sx, sy = self.world_to_screen(
            pos[0],
            pos[1],
        )

        role = enum_name(
            getattr(
                uav,
                "role",
                None,
            )
        )

        colour = role_colour(
            getattr(
                uav,
                "role",
                None,
            )
        )

        if getattr(
            uav,
            "failure_state",
            False,
        ):

            colour = FAILED

        # Follow/selection highlight.
        if (
            uid == self.selected_uav
            or uid == self.follow_uav
        ):

            pygame.draw.circle(
                self.screen,
                TEXT,
                (sx, sy),
                19,
                2,
            )

        # Altitude ring.
        altitude = float(
            pos[2]
        )

        radius = int(
            clamp(
                7
                + altitude * 0.03,
                8,
                13,
            )
        )

        pygame.draw.circle(
            self.screen,
            colour,
            (sx, sy),
            radius,
        )

        pygame.draw.circle(
            self.screen,
            TEXT,
            (sx, sy),
            radius,
            1,
        )

        # UAV heading.
        velocity = np.asarray(
            getattr(
                uav,
                "velocity",
                np.zeros(3),
            ),
            dtype=float,
        )

        speed = float(
            np.linalg.norm(
                velocity
            )
        )

        if speed > 0.1:

            direction = velocity[:2] / speed

            arrow_length = 23

            endpoint = (
                sx
                + int(
                    direction[0]
                    * arrow_length
                ),
                sy
                - int(
                    direction[1]
                    * arrow_length
                ),
            )

            self.line(
                (sx, sy),
                endpoint,
                colour,
                3,
            )

        # Velocity vector.
        if (
            self.show_vectors
            and speed > 0.1
        ):

            length = 30

            endpoint = (
                sx
                + int(
                    velocity[0]
                    / 5.0
                    * length
                ),
                sy
                - int(
                    velocity[1]
                    / 5.0
                    * length
                ),
            )

            self.line(
                (sx, sy),
                endpoint,
                WHITE,
                1,
            )

        if self.show_labels:

            label = (
                f"UAV-{uid}"
                f"  {role}"
            )

            self.text(
                label,
                sx + 17,
                sy - 12,
                colour,
                self.tiny,
            )

            if getattr(
                uav,
                "assigned_task_id",
                None,
            ) is not None:

                self.text(
                    f"→ P{uav.assigned_task_id}",
                    sx + 17,
                    sy + 3,
                    WHITE,
                    self.tiny,
                )

            self.text(
                f"{speed:.1f} m/s",
                sx + 17,
                sy + 17,
                MUTED,
                self.tiny,
            )

    def draw_uavs(self):

        self.update_trails()

        for uid, uav in self.engine.state.uavs.items():

            self.draw_trail(uid)

            self.draw_uav(
                uid,
                uav,
            )

    # ------------------------------------------------------------
    # Sidebar statistics
    # ------------------------------------------------------------

    def mission_stats(self):

        tasks = self.engine.state.tasks.values()

        completed = 0
        active = 0
        pending = 0
        expired = 0

        for task in tasks:

            status = enum_name(
                getattr(
                    task,
                    "status",
                    None,
                )
            )

            if status == "COMPLETED":
                completed += 1

            elif status == "EXPIRED":
                expired += 1

            elif status in {
                "ASSIGNED",
                "IN_PROGRESS",
            }:

                active += 1

            else:
                pending += 1

        return (
            completed,
            active,
            pending,
            expired,
        )

    def role_stats(self):

        result = {
            "RELAY": 0,
            "SURVEY": 0,
            "RESERVE": 0,
            "RETURN_HOME": 0,
            "FAILED": 0,
        }

        for uav in self.engine.state.uavs.values():

            role = enum_name(
                getattr(
                    uav,
                    "role",
                    None,
                )
            )

            if getattr(
                uav,
                "failure_state",
                False,
            ):

                result["FAILED"] += 1

            elif role in result:

                result[role] += 1

            elif role in {
                "LANDING",
                "RTH",
            }:

                result["RETURN_HOME"] += 1

            else:

                result["RESERVE"] += 1

        return result

    def network_stats(self):

        network = self.engine.state.network

        links = len(
            getattr(
                network,
                "links",
                {},
            )
        )

        analysis = getattr(
            self.engine,
            "network_analysis",
            None,
        )

        reachable = 0

        if analysis is not None:

            reachable = len(
                getattr(
                    analysis,
                    "gcs_reachable",
                    set(),
                )
            )

        return (
            links,
            reachable,
        )

    # ------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------

    def draw_progress_bar(
        self,
        x,
        y,
        width,
        height,
        fraction,
        colour,
    ):

        pygame.draw.rect(
            self.screen,
            PANEL_2,
            (x, y, width, height),
        )

        pygame.draw.rect(
            self.screen,
            colour,
            (
                x,
                y,
                int(
                    width
                    * clamp(
                        fraction,
                        0.0,
                        1.0,
                    )
                ),
                height,
            ),
        )

        pygame.draw.rect(
            self.screen,
            GRID,
            (x, y, width, height),
            1,
        )

    def draw_sidebar(self):

        pygame.draw.rect(
            self.screen,
            PANEL,
            (
                MAP_WIDTH,
                0,
                SIDEBAR,
                HEIGHT,
            ),
        )

        x = MAP_WIDTH + 22
        y = 18

        self.text(
            "UAV-X",
            x,
            y,
            TEXT,
            self.header,
        )

        y += 34

        self.text(
            "RESILIENT BVLOS SWARM",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 38

        sim_time = float(
            self.engine.state.timestamp_s
        )

        mission_duration = float(
            self.engine.config.simulation.mission_duration_s
        )

        fraction = (
            sim_time
            / max(
                mission_duration,
                1.0,
            )
        )

        self.text(
            "MISSION CLOCK",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 20

        self.text(
            f"{sim_time:07.1f} / "
            f"{mission_duration:.0f} s",
            x,
            y,
            TEXT,
            self.font,
        )

        y += 28

        self.draw_progress_bar(
            x,
            y,
            SIDEBAR - 45,
            8,
            fraction,
            RELAY,
        )

        y += 28

        self.text(
            f"SIM SPEED  {self.sim_speed:.1f}×",
            x,
            y,
            RELAY,
            self.small,
        )

        y += 21

        diagnostics = getattr(
            getattr(
                self.engine,
                "motion",
                None,
            ),
            "diagnostics",
            None,
        )

        if diagnostics is not None:

            self.text(
                f"MIN SEP    {diagnostics.min_separation_m:.1f} m",
                x,
                y,
                DONE
                if diagnostics.min_separation_m
                >= self.engine.config.uav.min_separation_m
                else FAILED,
                self.tiny,
            )

            y += 16

            self.text(
                f"MAX SPEED  {diagnostics.max_speed_mps:.2f} m/s",
                x,
                y,
                DONE
                if diagnostics.max_speed_mps
                <= self.engine.config.uav.max_speed_mps + 1e-6
                else FAILED,
                self.tiny,
            )

            y += 16

            self.text(
                f"SAFETY INT  {diagnostics.collision_interventions}",
                x,
                y,
                WARNING
                if diagnostics.collision_interventions
                else DONE,
                self.tiny,
            )

        y += 25

        # Mission.
        self.text(
            "MISSION",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 22

        completed, active, pending, expired = (
            self.mission_stats()
        )

        total = max(
            1,
            len(self.engine.state.pois),
        )

        self.text(
            f"Completed      {completed}/{total}",
            x,
            y,
            DONE,
        )

        y += 19

        self.text(
            f"Active         {active}",
            x,
            y,
            TEXT,
        )

        y += 19

        self.text(
            f"Pending        {pending}",
            x,
            y,
            POI,
        )

        y += 19

        self.text(
            f"Expired        {expired}",
            x,
            y,
            FAILED,
        )

        y += 33

        # Network.
        self.text(
            "NETWORK",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 22

        links, reachable = (
            self.network_stats()
        )

        self.text(
            f"Links          {links}",
            x,
            y,
            LINK,
        )

        y += 19

        self.text(
            f"GCS Reachable  "
            f"{reachable}/"
            f"{len(self.engine.state.uavs)}",
            x,
            y,
            DONE
            if reachable
            == len(
                self.engine.state.uavs
            )
            else WARNING,
        )

        y += 33

        # Roles.
        self.text(
            "SWARM ROLES",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 22

        roles = self.role_stats()

        for role in (
            "RELAY",
            "SURVEY",
            "RESERVE",
            "RETURN_HOME",
            "FAILED",
        ):

            self.text(
                f"{role:<12} "
                f"{roles[role]}",
                x,
                y,
                role_colour(role),
            )

            y += 18

        y += 21

        # Selected UAV.
        self.text(
            "SELECTED UAV",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 22

        if (
            self.selected_uav is not None
            and self.selected_uav
            in self.engine.state.uavs
        ):

            uav = self.engine.state.uavs[
                self.selected_uav
            ]

            pos = as_xyz(uav)

            vel = np.asarray(
                getattr(
                    uav,
                    "velocity",
                    np.zeros(3),
                ),
                dtype=float,
            )

            speed = float(
                np.linalg.norm(
                    vel
                )
            )

            self.text(
                f"UAV-{self.selected_uav}",
                x,
                y,
                TEXT,
                self.font,
            )

            y += 26

            self.text(
                f"Role       {enum_name(uav.role)}",
                x,
                y,
                role_colour(
                    uav.role
                ),
            )

            y += 18

            self.text(
                f"Position   "
                f"{pos[0]:.0f}, "
                f"{pos[1]:.0f}, "
                f"{pos[2]:.0f} m",
                x,
                y,
            )

            y += 18

            self.text(
                f"Speed      "
                f"{speed:.2f} m/s",
                x,
                y,
            )

            y += 18

            self.text(
                f"Comm       "
                f"{getattr(uav, 'communication_quality', 0.0):.2f}",
                x,
                y,
            )

            y += 18

            battery = getattr(
                getattr(
                    uav,
                    "battery",
                    None,
                ),
                "soc",
                0.0,
            )

            self.text(
                f"Battery    "
                f"{float(battery) * 100:.0f}%",
                x,
                y,
            )

            y += 30

        # Events.
        self.text(
            "EVENT STREAM",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 21

        for event in list(
            self.events
        )[-7:]:

            self.text(
                event,
                x,
                y,
                TEXT,
                self.tiny,
            )

            y += 17

        y = HEIGHT - 143

        self.text(
            "CONTROLS",
            x,
            y,
            MUTED,
            self.tiny,
        )

        y += 21

        controls = [
            "SPACE  Pause / Resume",
            "L      Links",
            "T      Trails",
            "V      Velocity vectors",
            "G      Grid",
            "TAB    Labels",
            "F      Follow selected UAV",
            "+/-    Simulation speed",
            "R      Reset",
            "ESC    Exit",
        ]

        for value in controls:

            self.text(
                value,
                x,
                y,
                MUTED,
                self.tiny,
            )

            y += 15

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    def handle_events(self):

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

                elif event.key == pygame.K_l:

                    self.show_links = (
                        not self.show_links
                    )

                elif event.key == pygame.K_t:

                    self.show_trails = (
                        not self.show_trails
                    )

                elif event.key == pygame.K_v:

                    self.show_vectors = (
                        not self.show_vectors
                    )

                elif event.key == pygame.K_g:

                    self.show_grid = (
                        not self.show_grid
                    )

                elif event.key == pygame.K_TAB:

                    self.show_labels = (
                        not self.show_labels
                    )

                elif event.key == pygame.K_f:

                    if self.selected_uav is None:

                        self.follow_uav = None

                    else:

                        if (
                            self.follow_uav
                            == self.selected_uav
                        ):

                            self.follow_uav = None

                        else:

                            self.follow_uav = (
                                self.selected_uav
                            )

                elif event.key in {
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                }:

                    self.sim_speed = min(
                        100.0,
                        self.sim_speed * 1.5,
                    )

                elif event.key == pygame.K_MINUS:

                    self.sim_speed = max(
                        0.25,
                        self.sim_speed / 1.5,
                    )

                elif (
                    pygame.K_0
                    <= event.key
                    <= pygame.K_9
                ):

                    value = (
                        event.key
                        - pygame.K_0
                    )

                    if value == 0:
                        self.selected_uav = None

                    elif (
                        value - 1
                        in self.engine.state.uavs
                    ):

                        self.selected_uav = (
                            value - 1
                        )

    # ------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------

    def apply_follow_camera(self):

        if (
            self.follow_uav is None
            or self.follow_uav
            not in self.engine.state.uavs
        ):
            return

        uav = self.engine.state.uavs[
            self.follow_uav
        ]

        pos = as_xyz(uav)

        # Smooth map shift is deliberately omitted;
        # the map is bounded to the complete operational area.
        _ = pos

    # ------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------

    def draw(self):

        self.screen.fill(BG)

        self.apply_follow_camera()

        self.draw_map()
        self.draw_network()
        self.draw_gcs()
        self.draw_pois()
        self.draw_uavs()
        self.draw_sidebar()

        pygame.display.flip()

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------

    def run(self):

        while self.running:

            real_dt = (
                self.clock.tick(FPS)
                / 1000.0
            )

            real_dt = clamp(
                real_dt,
                0.001,
                0.050,
            )

            self.handle_events()

            self.step_simulation(
                real_dt
            )

            self.detect_events()

            self.draw()

        pygame.quit()


def main():

    app = ProfessionalSimulator()

    app.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())

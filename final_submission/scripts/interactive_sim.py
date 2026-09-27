from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import pygame

from uavx.core import load_config
from uavx.simulation import SimulationEngine


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "official_stage1.yaml"

SCREEN_W = 1500
SCREEN_H = 900

PANEL_W = 340

MAP_W = SCREEN_W - PANEL_W
MAP_H = SCREEN_H

FPS = 60

DEFAULT_SIM_SPEED = 20.0
MIN_SIM_SPEED = 1.0
MAX_SIM_SPEED = 200.0

BACKGROUND = (9, 13, 20)
MAP_BG = (13, 19, 28)
GRID = (30, 41, 55)
PANEL_BG = (17, 24, 35)
TEXT = (225, 232, 240)
MUTED = (145, 158, 175)

GCS_COLOR = (255, 190, 60)
RELAY_COLOR = (50, 205, 255)
SURVEY_COLOR = (70, 235, 120)
RESERVE_COLOR = (170, 180, 195)
FAILED_COLOR = (255, 75, 85)
POI_COLOR = (255, 80, 190)
LINK_COLOR = (50, 180, 230)
TRAIL_COLOR = (70, 100, 125)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def enum_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if "." in text:
        text = text.split(".")[-1]
    return text.upper()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def position_of(obj):
    pos = getattr(obj, "position", None)

    if pos is None:
        return (0.0, 0.0, 0.0)

    try:
        return (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )
    except Exception:
        return (0.0, 0.0, 0.0)


def battery_percent(uav) -> float:
    battery = getattr(uav, "battery", None)

    if battery is None:
        return 100.0

    candidates = (
        "soc",
        "state_of_charge",
        "charge_fraction",
        "remaining_fraction",
    )

    for name in candidates:
        if hasattr(battery, name):
            value = safe_float(getattr(battery, name), 1.0)

            if value <= 1.0:
                return max(0.0, min(100.0, value * 100.0))

            return max(0.0, min(100.0, value))

    return 100.0


def task_status(task) -> str:
    return enum_text(getattr(task, "status", None))


def role_color(role: str):
    role = enum_text(role)

    if "FAILED" in role:
        return FAILED_COLOR

    if "RELAY" in role:
        return RELAY_COLOR

    if "SURVEY" in role:
        return SURVEY_COLOR

    if "RESERVE" in role:
        return RESERVE_COLOR

    return TEXT


# ---------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------

class UAVXViewer:

    def __init__(self):

        pygame.init()

        pygame.display.set_caption(
            "UAV-X Resilient BVLOS Swarm"
        )

        self.screen = pygame.display.set_mode(
            (SCREEN_W, SCREEN_H)
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "DejaVu Sans",
            18,
        )

        self.small_font = pygame.font.SysFont(
            "DejaVu Sans",
            14,
        )

        self.title_font = pygame.font.SysFont(
            "DejaVu Sans",
            24,
            bold=True,
        )

        self.sim_speed = DEFAULT_SIM_SPEED

        self.paused = False

        self.show_links = True
        self.show_trails = True

        self.running = True

        self.engine = None

        self.trails = {}
        self.prev_roles = {}
        self.prev_tasks = {}

        self.event_log = deque(maxlen=8)

        self.reset()

    # -----------------------------------------------------------------

    def reset(self):

        config = load_config(str(CONFIG))

        self.engine = SimulationEngine.from_config(
            config
        )

        self.engine.start()

        self.trails.clear()
        self.prev_roles.clear()
        self.prev_tasks.clear()
        self.event_log.clear()

        self.log_event(
            "Simulation initialized"
        )

    # -----------------------------------------------------------------

    def log_event(self, message):

        timestamp = safe_float(
            getattr(
                self.engine.state,
                "timestamp_s",
                0.0,
            )
        )

        self.event_log.append(
            f"{timestamp:06.1f}s  {message}"
        )

    # -----------------------------------------------------------------

    def map_to_screen(self, x, y):

        x = max(
            0.0,
            min(
                x,
                self.engine.config.environment.width_m,
            ),
        )

        y = max(
            0.0,
            min(
                y,
                self.engine.config.environment.height_m,
            ),
        )

        sx = int(
            x /
            self.engine.config.environment.width_m
            * (MAP_W - 40)
        ) + 20

        sy = MAP_H - (
            int(
                y /
                self.engine.config.environment.height_m
                * (MAP_H - 40)
            ) + 20
        )

        return sx, sy

    # -----------------------------------------------------------------

    def draw_text(
        self,
        text,
        x,
        y,
        color=TEXT,
        font=None,
    ):

        if font is None:
            font = self.font

        surface = font.render(
            str(text),
            True,
            color,
        )

        self.screen.blit(
            surface,
            (x, y),
        )

    # -----------------------------------------------------------------

    def draw_grid(self):

        self.screen.fill(BACKGROUND)

        pygame.draw.rect(
            self.screen,
            MAP_BG,
            (0, 0, MAP_W, MAP_H),
        )

        step = 100

        for x in range(
            0,
            int(
                self.engine.config.environment.width_m
            ) + 1,
            step,
        ):

            sx, _ = self.map_to_screen(
                x,
                0,
            )

            pygame.draw.line(
                self.screen,
                GRID,
                (sx, 20),
                (sx, MAP_H - 20),
                1,
            )

            self.draw_text(
                f"{x}m",
                sx - 14,
                MAP_H - 18,
                MUTED,
                self.small_font,
            )

        for y in range(
            0,
            int(
                self.engine.config.environment.height_m
            ) + 1,
            step,
        ):

            _, sy = self.map_to_screen(
                0,
                y,
            )

            pygame.draw.line(
                self.screen,
                GRID,
                (20, sy),
                (MAP_W - 20, sy),
                1,
            )

            self.draw_text(
                f"{y}m",
                2,
                sy - 7,
                MUTED,
                self.small_font,
            )

    # -----------------------------------------------------------------

    def extract_links(self):

        network = getattr(
            self.engine.state,
            "network",
            None,
        )

        if network is None:
            return []

        links = []

        raw_edges = getattr(
            network,
            "edges",
            None,
        )

        if raw_edges is not None:

            try:

                for edge in raw_edges:

                    if isinstance(edge, (tuple, list)):
                        if len(edge) >= 2:
                            links.append(
                                (
                                    edge[0],
                                    edge[1],
                                )
                            )

            except Exception:
                pass

        raw_links = getattr(
            network,
            "links",
            None,
        )

        if raw_links:

            try:

                for key in raw_links.keys():

                    if isinstance(
                        key,
                        (tuple, list),
                    ) and len(key) >= 2:

                        links.append(
                            (
                                key[0],
                                key[1],
                            )
                        )

            except Exception:
                pass

        unique = set()

        for a, b in links:

            key = tuple(
                sorted(
                    (
                        str(a),
                        str(b),
                    )
                )
            )

            if key in unique:
                continue

            unique.add(key)

        return links

    # -----------------------------------------------------------------

    def draw_links(self):

        if not self.show_links:
            return

        links = self.extract_links()

        positions = {
            str(
                uav_id
            ): position_of(uav)
            for uav_id, uav in self.engine.state.uavs.items()
        }

        gcs_position = getattr(
            self.engine,
            "gcs_position",
            (0.0, 0.0, 20.0),
        )

        gcs_position = (
            float(gcs_position[0]),
            float(gcs_position[1]),
            float(gcs_position[2]),
        )

        for a, b in links:

            a_key = str(a)
            b_key = str(b)

            if a_key not in positions and a_key != "GCS":
                continue

            if b_key not in positions and b_key != "GCS":
                continue

            if a_key == "GCS":
                pa = gcs_position
            else:
                pa = positions[a_key]

            if b_key == "GCS":
                pb = gcs_position
            else:
                pb = positions[b_key]

            ax, ay = self.map_to_screen(
                pa[0],
                pa[1],
            )

            bx, by = self.map_to_screen(
                pb[0],
                pb[1],
            )

            pygame.draw.line(
                self.screen,
                LINK_COLOR,
                (ax, ay),
                (bx, by),
                2,
            )

    # -----------------------------------------------------------------

    def draw_gcs(self):

        gcs = getattr(
            self.engine,
            "gcs_position",
            (0.0, 0.0, 20.0),
        )

        sx, sy = self.map_to_screen(
            gcs[0],
            gcs[1],
        )

        pygame.draw.circle(
            self.screen,
            GCS_COLOR,
            (sx, sy),
            10,
        )

        pygame.draw.circle(
            self.screen,
            TEXT,
            (sx, sy),
            13,
            2,
        )

        self.draw_text(
            "GCS",
            sx + 16,
            sy - 10,
            GCS_COLOR,
            self.small_font,
        )

    # -----------------------------------------------------------------

    def draw_pois(self):

        tasks = getattr(
            self.engine.state,
            "tasks",
            {},
        )

        pois = getattr(
            self.engine.state,
            "pois",
            {},
        )

        for poi_id, poi in pois.items():

            x, y, _ = position_of(poi)

            sx, sy = self.map_to_screen(
                x,
                y,
            )

            status = ""

            task = tasks.get(poi_id)

            if task is not None:
                status = task_status(task)

            if "COMPLETED" in status:

                color = SURVEY_COLOR

            elif (
                "EXPIRED" in status or
                "FAILED" in status
            ):

                color = FAILED_COLOR

            else:

                color = POI_COLOR

            pygame.draw.circle(
                self.screen,
                color,
                (sx, sy),
                7,
            )

            pygame.draw.circle(
                self.screen,
                color,
                (sx, sy),
                11,
                1,
            )

            self.draw_text(
                f"PoI-{poi_id}",
                sx + 12,
                sy - 7,
                color,
                self.small_font,
            )

    # -----------------------------------------------------------------

    def draw_uavs(self):

        for uav_id, uav in self.engine.state.uavs.items():

            x, y, _ = position_of(uav)

            sx, sy = self.map_to_screen(
                x,
                y,
            )

            role = enum_text(
                getattr(
                    uav,
                    "role",
                    "UNKNOWN",
                )
            )

            color = role_color(role)

            radius = 9

            if "RELAY" in role:
                radius = 11

            pygame.draw.circle(
                self.screen,
                color,
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

            battery = battery_percent(uav)

            label = (
                f"UAV-{uav_id} "
                f"{role} "
                f"{battery:.0f}%"
            )

            self.draw_text(
                label,
                sx + 14,
                sy - 9,
                color,
                self.small_font,
            )

            if self.show_trails:

                trail = self.trails.setdefault(
                    uav_id,
                    deque(maxlen=80),
                )

                trail.append(
                    (
                        sx,
                        sy,
                    )
                )

                if len(trail) > 1:

                    pygame.draw.lines(
                        self.screen,
                        TRAIL_COLOR,
                        False,
                        list(trail),
                        1,
                    )

    # -----------------------------------------------------------------

    def task_counts(self):

        tasks = getattr(
            self.engine.state,
            "tasks",
            {},
        )

        completed = 0
        active = 0
        pending = 0
        expired = 0

        for task in tasks.values():

            status = task_status(task)

            if "COMPLETED" in status:

                completed += 1

            elif "EXPIRED" in status:

                expired += 1

            elif (
                "ACTIVE" in status or
                "ASSIGNED" in status
            ):

                active += 1

            else:

                pending += 1

        return (
            completed,
            active,
            pending,
            expired,
        )

    # -----------------------------------------------------------------

    def role_counts(self):

        counts = {
            "RELAY": 0,
            "SURVEY": 0,
            "RESERVE": 0,
            "FAILED": 0,
        }

        for uav in self.engine.state.uavs.values():

            role = enum_text(
                getattr(
                    uav,
                    "role",
                    "",
                )
            )

            if "FAILED" in role:
                counts["FAILED"] += 1

            elif "RELAY" in role:
                counts["RELAY"] += 1

            elif "SURVEY" in role:
                counts["SURVEY"] += 1

            else:
                counts["RESERVE"] += 1

        return counts

    # -----------------------------------------------------------------

    def network_stats(self):

        links = self.extract_links()

        nodes = len(
            getattr(
                self.engine.state,
                "uavs",
                {},
            )
        )

        return (
            nodes,
            len(links),
        )

    # -----------------------------------------------------------------

    def draw_panel(self):

        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (MAP_W, 0, PANEL_W, SCREEN_H),
        )

        x = MAP_W + 24
        y = 22

        sim_time = safe_float(
            getattr(
                self.engine.state,
                "timestamp_s",
                0.0,
            )
        )

        mission_duration = safe_float(
            getattr(
                self.engine.config.simulation,
                "mission_duration_s",
                2700.0,
            )
        )

        completed, active, pending, expired = (
            self.task_counts()
        )

        role_counts = self.role_counts()

        nodes, links = self.network_stats()

        self.draw_text(
            "UAV-X",
            x,
            y,
            TEXT,
            self.title_font,
        )

        self.draw_text(
            "RESILIENT BVLOS SWARM",
            x,
            y + 32,
            MUTED,
            self.small_font,
        )

        y += 78

        self.draw_text(
            "SIMULATION",
            x,
            y,
            MUTED,
            self.small_font,
        )

        y += 22

        self.draw_text(
            f"{sim_time:07.1f} / {mission_duration:.0f} s",
            x,
            y,
            TEXT,
            self.font,
        )

        y += 28

        self.draw_text(
            f"Speed: {self.sim_speed:.0f}x",
            x,
            y,
            RELAY_COLOR,
            self.small_font,
        )

        y += 42

        self.draw_text(
            "MISSION",
            x,
            y,
            MUTED,
            self.small_font,
        )

        y += 24

        self.draw_text(
            f"Completed     {completed}",
            x,
            y,
            SURVEY_COLOR,
            self.small_font,
        )

        y += 21

        self.draw_text(
            f"Active        {active}",
            x,
            y,
            TEXT,
            self.small_font,
        )

        y += 21

        self.draw_text(
            f"Pending       {pending}",
            x,
            y,
            POI_COLOR,
            self.small_font,
        )

        y += 21

        self.draw_text(
            f"Expired       {expired}",
            x,
            y,
            FAILED_COLOR,
            self.small_font,
        )

        y += 42

        self.draw_text(
            "SWARM",
            x,
            y,
            MUTED,
            self.small_font,
        )

        y += 24

        self.draw_text(
            f"Relay         {role_counts['RELAY']}",
            x,
            y,
            RELAY_COLOR,
            self.small_font,
        )

        y += 21

        self.draw_text(
            f"Survey        {role_counts['SURVEY']}",
            x,
            y,
            SURVEY_COLOR,
            self.small_font,
        )

        y += 21

        self.draw_text(
            f"Reserve       {role_counts['RESERVE']}",
            x,
            y,
            RESERVE_COLOR,
            self.small_font,
        )

        y += 21

        self.draw_text(
            f"Failed        {role_counts['FAILED']}",
            x,
            y,
            FAILED_COLOR,
            self.small_font,
        )

        y += 42

        self.draw_text(
            "NETWORK",
            x,
            y,
            MUTED,
            self.small_font,
        )

        y += 24

        self.draw_text(
            f"UAV nodes     {nodes}",
            x,
            y,
            TEXT,
            self.small_font,
        )

        y += 21

        self.draw_text(
            f"Links         {links}",
            x,
            y,
            RELAY_COLOR,
            self.small_font,
        )

        y += 42

        self.draw_text(
            "EVENT LOG",
            x,
            y,
            MUTED,
            self.small_font,
        )

        y += 23

        for event in list(
            self.event_log
        )[-7:]:

            self.draw_text(
                event,
                x,
                y,
                TEXT,
                self.small_font,
            )

            y += 21

        y = SCREEN_H - 125

        self.draw_text(
            "CONTROLS",
            x,
            y,
            MUTED,
            self.small_font,
        )

        y += 22

        controls = (
            "SPACE  pause / resume",
            "+/-    simulation speed",
            "L      communication links",
            "T      UAV trails",
            "R      reset",
            "ESC    quit",
        )

        for line in controls:

            self.draw_text(
                line,
                x,
                y,
                TEXT,
                self.small_font,
            )

            y += 19

    # -----------------------------------------------------------------

    def detect_events(self):

        for uav_id, uav in self.engine.state.uavs.items():

            role = enum_text(
                getattr(
                    uav,
                    "role",
                    "",
                )
            )

            previous = self.prev_roles.get(
                uav_id
            )

            if (
                previous is not None and
                role != previous
            ):

                self.log_event(
                    f"UAV-{uav_id} "
                    f"{previous} -> {role}"
                )

            self.prev_roles[uav_id] = role

        tasks = getattr(
            self.engine.state,
            "tasks",
            {},
        )

        for task_id, task in tasks.items():

            status = task_status(task)

            previous = self.prev_tasks.get(
                task_id
            )

            if (
                previous is not None and
                status != previous
            ):

                if (
                    "COMPLETED" in status or
                    "EXPIRED" in status
                ):

                    self.log_event(
                        f"Task {task_id} -> {status}"
                    )

            self.prev_tasks[task_id] = status

    # -----------------------------------------------------------------

    def step_simulation(self):

        if self.paused:
            return

        # Run several simulation steps per rendered frame.
        sim_dt = 0.1
        steps = max(
            1,
            int(
                self.sim_speed *
                0.25
            ),
        )

        for _ in range(steps):

            if (
                self.engine.state.mission.status
                != getattr(
                    self.engine.state.mission,
                    "status",
                    None,
                )
            ):
                pass

            self.engine.step(
                sim_dt
            )

            if (
                self.engine.state.mission.status
                == "COMPLETED"
            ):
                self.paused = True
                self.log_event(
                    "Mission completed"
                )
                break

            if safe_float(
                getattr(
                    self.engine.state,
                    "timestamp_s",
                    0.0,
                )
            ) >= safe_float(
                getattr(
                    self.engine.config.simulation,
                    "mission_duration_s",
                    2700.0,
                )
            ):

                self.paused = True

                self.log_event(
                    "Mission duration reached"
                )

                break

    # -----------------------------------------------------------------

    def draw(self):

        self.draw_grid()

        self.draw_links()
        self.draw_gcs()
        self.draw_pois()
        self.draw_uavs()
        self.draw_panel()

        pygame.display.flip()

    # -----------------------------------------------------------------

    def handle_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                self.running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:

                    self.running = False

                elif event.key == pygame.K_SPACE:

                    self.paused = not self.paused

                    self.log_event(
                        "PAUSED"
                        if self.paused
                        else "RESUMED"
                    )

                elif event.key == pygame.K_r:

                    self.reset()

                elif event.key in (
                    pygame.K_PLUS,
                    pygame.K_EQUALS,
                ):

                    self.sim_speed = min(
                        MAX_SIM_SPEED,
                        self.sim_speed * 2,
                    )

                    self.log_event(
                        f"Speed {self.sim_speed:.0f}x"
                    )

                elif event.key == pygame.K_MINUS:

                    self.sim_speed = max(
                        MIN_SIM_SPEED,
                        self.sim_speed / 2,
                    )

                    self.log_event(
                        f"Speed {self.sim_speed:.0f}x"
                    )

                elif event.key == pygame.K_l:

                    self.show_links = (
                        not self.show_links
                    )

                elif event.key == pygame.K_t:

                    self.show_trails = (
                        not self.show_trails
                    )

    # -----------------------------------------------------------------

    def run(self):

        while self.running:

            self.handle_events()
            self.step_simulation()
            self.detect_events()
            self.draw()

            self.clock.tick(FPS)

        pygame.quit()


def main():

    viewer = UAVXViewer()

    viewer.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())

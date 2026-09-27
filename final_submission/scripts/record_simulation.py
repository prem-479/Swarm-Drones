import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from uavx.core import load_config
from uavx.simulation import SimulationEngine


CONFIG = "configs/official_stage1.yaml"
OUTPUT = "results/uavx_stage1_simulation.mp4"

config = load_config(CONFIG)
engine = SimulationEngine.from_config(config)
engine.start()

FRAME_INTERVAL_S = 5.0
FPS = 10

total_frames = (
    int(config.simulation.mission_duration_s / FRAME_INTERVAL_S) + 1
)

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_xlim(
    0,
    config.environment.width_m,
)
ax.set_ylim(
    0,
    config.environment.height_m,
)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("UAV-X Resilient BVLOS Swarm — Stage 1")
ax.grid(True, alpha=0.25)

gcs_plot, = ax.plot(
    [0],
    [0],
    marker="*",
    markersize=18,
    linestyle="None",
    label="GCS",
)

uav_plot, = ax.plot(
    [],
    [],
    marker="o",
    markersize=7,
    linestyle="None",
    label="UAV",
)

survey_plot, = ax.plot(
    [],
    [],
    marker="o",
    markersize=9,
    linestyle="None",
    label="Survey",
)

relay_plot, = ax.plot(
    [],
    [],
    marker="s",
    markersize=8,
    linestyle="None",
    label="Relay",
)

poi_plot, = ax.plot(
    [],
    [],
    marker="x",
    markersize=9,
    linestyle="None",
    label="PoI",
)

time_text = ax.text(
    0.02,
    0.97,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=11,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
    ),
)

info_text = ax.text(
    0.02,
    0.88,
    "",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85,
    ),
)

ax.legend(loc="upper right")

link_lines = []


def clear_links():
    global link_lines

    for line in link_lines:
        line.remove()

    link_lines = []


def update(frame):

    target_time = min(
        frame * FRAME_INTERVAL_S,
        config.simulation.mission_duration_s,
    )

    while engine.state.timestamp_s < target_time:
        remaining = target_time - engine.state.timestamp_s

        dt = min(
            config.simulation.timestep_s,
            remaining,
        )

        if dt <= 0:
            break

        engine.step(dt)

    state = engine.state

    active = [
        u
        for u in state.uavs.values()
        if not u.failure_state
    ]

    survey = [
        u
        for u in active
        if u.role.value == "SURVEY"
    ]

    relays = [
        u
        for u in active
        if u.role.value == "RELAY"
    ]

    # --------------------------------------------------------
    # UAV positions
    # --------------------------------------------------------
    uav_plot.set_data(
        [float(u.position[0]) for u in active],
        [float(u.position[1]) for u in active],
    )

    survey_plot.set_data(
        [float(u.position[0]) for u in survey],
        [float(u.position[1]) for u in survey],
    )

    relay_plot.set_data(
        [float(u.position[0]) for u in relays],
        [float(u.position[1]) for u in relays],
    )

    # --------------------------------------------------------
    # PoIs
    # --------------------------------------------------------
    visible_pois = [
        p
        for p in state.pois.values()
        if getattr(p.status, "value", p.status)
        not in {"COMPLETED", "EXPIRED"}
    ]

    poi_plot.set_data(
        [float(p.position[0]) for p in visible_pois],
        [float(p.position[1]) for p in visible_pois],
    )

    # --------------------------------------------------------
    # Communication graph
    # --------------------------------------------------------
    clear_links()

    positions = {
        int(u.uav_id): np.asarray(
            u.position,
            dtype=float,
        )
        for u in active
    }

    positions[-1] = np.array(
        [0.0, 0.0, 20.0],
        dtype=float,
    )

    for a, b in state.network.edges:

        if a not in positions or b not in positions:
            continue

        pa = positions[a]
        pb = positions[b]

        line, = ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            linewidth=0.7,
            alpha=0.25,
        )

        link_lines.append(line)

    # --------------------------------------------------------
    # Mission statistics
    # --------------------------------------------------------
    completed = sum(
        1
        for task in state.tasks.values()
        if getattr(task.status, "value", task.status)
        == "COMPLETED"
    )

    expired = sum(
        1
        for task in state.tasks.values()
        if getattr(task.status, "value", task.status)
        == "EXPIRED"
    )

    partitioned = False

    try:
        network = state.network

        nodes = set(network.nodes)
        adjacency = {n: set() for n in nodes}

        for a, b in network.edges:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

        components = 0
        unseen = set(nodes)

        while unseen:
            components += 1

            stack = [unseen.pop()]

            while stack:
                node = stack.pop()

                for neighbor in adjacency.get(node, set()):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)

        partitioned = components > 1

    except Exception:
        partitioned = False

    time_text.set_text(
        f"Simulation Time: "
        f"{state.timestamp_s:7.1f} / "
        f"{config.simulation.mission_duration_s:.0f} s"
    )

    info_text.set_text(
        f"PoIs: {len(state.pois)}    "
        f"Completed: {completed}    "
        f"Expired: {expired}\n"
        f"Survey: {len(survey)}    "
        f"Relays: {len(relays)}    "
        f"Network Partitioned: {partitioned}\n"
        f"Network Links: {len(state.network.edges)}"
    )

    return (
        uav_plot,
        survey_plot,
        relay_plot,
        poi_plot,
        time_text,
        info_text,
        *link_lines,
    )


print("=" * 70)
print("UAV-X SIMULATION VIDEO")
print("=" * 70)
print(f"Area             : {config.environment.width_m} x "
      f"{config.environment.height_m} m")
print(f"Mission duration : {config.simulation.mission_duration_s:.0f} s")
print(f"Frame interval   : {FRAME_INTERVAL_S} s")
print(f"Video FPS        : {FPS}")
print(f"Frames           : {total_frames}")
print(f"Output           : {OUTPUT}")
print()

animation = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=1000 / FPS,
    blit=False,
    repeat=False,
)

writer = FFMpegWriter(
    fps=FPS,
    bitrate=5000,
    metadata={
        "title": "UAV-X Stage-1 Simulation",
        "artist": "UAV-X Resilient Swarm",
    },
)

animation.save(
    OUTPUT,
    writer=writer,
    dpi=120,
)

plt.close(fig)

print()
print("=" * 70)
print("VIDEO CREATED")
print("=" * 70)
print(OUTPUT)

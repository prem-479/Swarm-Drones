from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class UAVRole(str, Enum):
    IDLE = "IDLE"
    SURVEY = "SURVEY"
    RELAY = "RELAY"
    TRANSIT = "TRANSIT"
    RECOVERY = "RECOVERY"
    RESERVE = "RESERVE"
    RETURN_HOME = "RETURN_HOME"
    LANDING = "LANDING"
    FAILED = "FAILED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class MissionStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FailureType(str, Enum):
    UAV_FAILURE = "UAV_FAILURE"
    LINK_FAILURE = "LINK_FAILURE"
    SENSOR_FAILURE = "SENSOR_FAILURE"
    COMMUNICATION_DEGRADATION = "COMMUNICATION_DEGRADATION"


class LinkStatus(str, Enum):
    UP = "UP"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


@dataclass
class BatteryState:
    soc: float = 1.0
    flight_time_s: float = 0.0
    energy_consumed: float = 0.0
    rth_triggered: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.soc <= 1.0:
            raise ValueError("Battery SOC must be in [0, 1]")


@dataclass
class UAVState:
    uav_id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    role: UAVRole
    battery: BatteryState
    mission_state: str = "IDLE"
    assigned_task_id: Optional[int] = None
    target: Optional[np.ndarray] = None
    neighbors: set[int] = field(default_factory=set)
    communication_quality: float = 1.0
    failure_state: bool = False
    home_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    flight_time_s: float = 0.0
    distance_travelled_m: float = 0.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.acceleration = np.asarray(self.acceleration, dtype=float)
        self.home_position = np.asarray(self.home_position, dtype=float)

        for vector_name, vector in (
            ("position", self.position),
            ("velocity", self.velocity),
            ("acceleration", self.acceleration),
            ("home_position", self.home_position),
        ):
            if vector.shape != (3,):
                raise ValueError(f"{vector_name} must have shape (3,)")

        if not 0.0 <= self.communication_quality <= 1.0:
            raise ValueError("communication_quality must be in [0, 1]")


@dataclass
class PoI:
    poi_id: int
    position: np.ndarray
    priority: float
    spawn_time_s: float
    discovery_time_s: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    reporting_deadline_s: float = 10.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)

        if self.position.shape != (3,):
            raise ValueError("PoI position must have shape (3,)")

        if self.priority < 0:
            raise ValueError("PoI priority cannot be negative")


@dataclass
class Task:
    task_id: int
    poi_id: int
    priority: float
    created_time_s: float
    status: TaskStatus = TaskStatus.PENDING
    assigned_uav_id: Optional[int] = None
    completion_time_s: Optional[float] = None
    version: int = 0


@dataclass
class CommunicationLink:
    source_id: int
    target_id: int
    distance_m: float
    link_quality: float
    packet_delivery_probability: float
    latency_ms: float
    availability: bool
    status: LinkStatus = LinkStatus.UP

    def __post_init__(self) -> None:
        if self.distance_m < 0:
            raise ValueError("Link distance cannot be negative")
        if not 0.0 <= self.link_quality <= 1.0:
            raise ValueError("link_quality must be in [0, 1]")
        if not 0.0 <= self.packet_delivery_probability <= 1.0:
            raise ValueError(
                "packet_delivery_probability must be in [0, 1]"
            )


@dataclass
class NetworkGraph:
    timestamp_s: float
    nodes: list[int]
    edges: set[tuple[int, int]]
    links: dict[tuple[int, int], CommunicationLink] = field(
        default_factory=dict
    )
    laplacian: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )

    def update_laplacian(self) -> None:
        n = len(self.nodes)
        self.laplacian = np.zeros((n, n), dtype=float)

        node_index = {node_id: index for index, node_id in enumerate(self.nodes)}

        for source, target in self.edges:
            if source not in node_index or target not in node_index:
                continue

            i = node_index[source]
            j = node_index[target]

            self.laplacian[i, j] = -1.0
            self.laplacian[j, i] = -1.0
            self.laplacian[i, i] += 1.0
            self.laplacian[j, j] += 1.0

    def degree(self, node_id: int) -> int:
        return sum(node_id in edge for edge in self.edges)


@dataclass
class MissionState:
    mission_id: str
    status: MissionStatus
    current_time_s: float = 0.0
    completed_pois: int = 0
    missed_pois: int = 0
    active_tasks: set[int] = field(default_factory=set)
    discovered_pois: set[int] = field(default_factory=set)


@dataclass
class FailureEvent:
    event_id: str
    timestamp_s: float
    failure_type: FailureType
    target_id: int
    duration_s: Optional[float] = None
    description: str = ""


@dataclass
class RoleState:
    uav_id: int
    current_role: UAVRole
    previous_role: Optional[UAVRole] = None
    changed_at_s: float = 0.0
    locked_until_s: Optional[float] = None


@dataclass
class Trajectory:
    uav_id: int
    timestamps_s: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray

    def __post_init__(self) -> None:
        self.timestamps_s = np.asarray(self.timestamps_s, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")

        if self.velocities.ndim != 2 or self.velocities.shape[1] != 3:
            raise ValueError("velocities must have shape (N, 3)")


@dataclass
class MetricSnapshot:
    timestamp_s: float
    pois_completed: int = 0
    completion_rate: float = 0.0
    pdr: float = 0.0
    average_latency_ms: float = 0.0
    worst_latency_ms: float = 0.0
    connectivity_available: float = 0.0
    communication_downtime_s: float = 0.0
    relay_reallocations: int = 0
    recovery_time_s: float = 0.0
    collision_count: int = 0
    minimum_separation_m: float = float("inf")
    geofence_violations: int = 0
    battery_depletion_events: int = 0
    distance_travelled_m: float = 0.0
    role_changes: int = 0
    controller_time_ms: float = 0.0


@dataclass
class SimulationState:
    timestamp_s: float
    uavs: dict[int, UAVState]
    pois: dict[int, PoI]
    tasks: dict[int, Task]
    network: NetworkGraph
    mission: MissionState
    failures: list[FailureEvent] = field(default_factory=list)
    roles: dict[int, RoleState] = field(default_factory=dict)
    metrics: MetricSnapshot | None = None

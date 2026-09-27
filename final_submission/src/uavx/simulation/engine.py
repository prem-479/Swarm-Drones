from __future__ import annotations

from time import perf_counter

import numpy as np

from uavx.communication import GCS_NODE_ID, CommunicationEngine
from uavx.autonomy import SwarmDecisionEngine
from uavx.network import NetworkGraphAnalyzer
from uavx.core.config import UAVXConfig
from uavx.core.events import EventQueue, EventType, SimulationEvent
from uavx.core.models import (
    BatteryState,
    MissionState,
    MissionStatus,
    NetworkGraph,
    RoleState,
    SimulationState,
    UAVRole,
    UAVState,
)
from uavx.scenario.generator import ScenarioGenerator
from uavx.simulation.professional_motion import ProfessionalMotionSystem
from uavx.simulation.dynamics import HighFidelityFlightDynamics


class SimulationEngine:
    """
    Custom 2D/2.5D simulation engine.

    High-frequency:
        kinematic updates

    Event-driven:
        scenario events and later mission/network/failure events

    Medium-frequency network updates will be added in a later phase.
    """

    def __init__(
        self,
        config: UAVXConfig,
        scenario_generator: ScenarioGenerator,
    ) -> None:
        self.config = config
        self.scenario_generator = scenario_generator
        self.events = EventQueue()

        self.communication = CommunicationEngine(
            config=config.communication,
            seed=config.simulation.seed,
        )

        # Engineering GCS position for the initial simulator.
        # GCS is deliberately outside the 1000 x 1000 m
        # operational area, as required by the challenge.
        self.gcs_position = np.array(
            [-50.0, 500.0, 25.0],
            dtype=float,
        )

        # Communication is a medium-frequency subsystem.
        self._network_update_interval_s = 1.0
        self._next_network_update_s = 0.0
        self.network_analysis = None

        # Professional high-frequency flight/safety layer.
        self.motion = ProfessionalMotionSystem(
            max_speed_mps=config.uav.max_speed_mps,
            max_acceleration_mps2=config.uav.max_acceleration_mps2,
            min_separation_m=config.uav.min_separation_m,
            width_m=config.environment.width_m,
            height_m=config.environment.height_m,
            max_altitude_m=config.environment.max_altitude_m,
        )

        # High-frequency physics stays at the official 0.1 s
        # simulation timestep. Swarm decisions run at a lower
        # medium-frequency rate to avoid unnecessary recomputation.
        self._decision_update_interval_s = 0.5
        self._next_decision_update_s = 0.0

        self.swarm = SwarmDecisionEngine(
            config=config,
            gcs_position=self.gcs_position,
        )

        self.flight_dynamics = HighFidelityFlightDynamics()

        self.state = SimulationState(
            timestamp_s=0.0,
            uavs={},
            pois={},
            tasks={},
            network=NetworkGraph(
                timestamp_s=0.0,
                nodes=[],
                edges=set(),
            ),
            mission=MissionState(
                mission_id="uavx-stage1",
                status=MissionStatus.READY,
            ),
        )

        self._initialize_uavs()

        # Stable connected launch formation.
        #
        # 2 staggered rows keep every aircraft above the
        # 20 m separation boundary while maintaining a
        # connected starting topology to the external GCS.
        columns = 5
        spacing_x = 58.0
        spacing_y = 62.0

        for uav_id, uav in self.state.uavs.items():

            row = uav_id // columns
            column = uav_id % columns

            y_offset = (
                spacing_y * 0.5
                if row % 2
                else 0.0
            )

            uav.position[:] = np.array(
                [
                    40.0 + column * spacing_x,
                    470.0 + row * spacing_y + y_offset,
                    25.0,
                ],
                dtype=float,
            )

            uav.home_position[:] = (
                uav.position
            )

        self.motion.reserve_slots = {
            uid: uav.position.copy()
            for uid, uav
            in self.state.uavs.items()
        }
        self._update_communication_graph(0.0)
        self._schedule_scenario()

    @classmethod
    def from_config(cls, config: UAVXConfig) -> SimulationEngine:
        scenario_generator = ScenarioGenerator(
            width_m=config.environment.width_m,
            height_m=config.environment.height_m,
            max_pois=config.scenario.poi_count,
            reporting_deadline_s=config.mission.poi_reporting_deadline_s,
            seed=config.simulation.seed,
        )

        return cls(config, scenario_generator)

    def _initialize_uavs(self) -> None:
        """
        Initial positions are an engineering initialization only.
        They are deliberately kept simple until the swarm deployment
        and relay-placement subsystem is implemented.
        """
        count = self.config.uav.count

        # Communication-aware launch formation.
        #
        # The GCS sits outside the operational area at x < 0.
        # UAVs begin in two rows close enough to maintain a
        # connected launch topology while preserving the official
        # 20 m minimum separation.
        #
        # This is an initialization formation, not a hard-coded
        # mission route.
        columns = min(5, max(1, count))
        spacing_x = 35.0
        spacing_y = 70.0

        launch_x = 30.0
        launch_y = 465.0
        launch_altitude = 25.0

        for uav_id in range(count):

            row = uav_id // columns
            column = uav_id % columns

            position = np.array(
                [
                    launch_x + column * spacing_x,
                    launch_y + row * spacing_y,
                    launch_altitude,
                ],
                dtype=float,
            )

            home = position.copy()

            self.state.uavs[uav_id] = UAVState(
                uav_id=uav_id,
                position=position,
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                role=UAVRole.RESERVE,
                battery=BatteryState(),
                home_position=home,
            )

            self.state.roles[uav_id] = RoleState(
                uav_id=uav_id,
                current_role=UAVRole.RESERVE,
                changed_at_s=0.0,
            )

        self.state.network.nodes = list(self.state.uavs.keys())
        self.state.network.update_laplacian()

    def _schedule_scenario(self) -> None:
        self.scenario_generator.schedule_poi_events(
            event_queue=self.events,
            mission_duration_s=self.config.simulation.mission_duration_s,
        )

        self.events.push(
            timestamp_s=self.config.simulation.mission_duration_s,
            event_type=EventType.MISSION_END,
        )

    def start(self) -> None:
        if self.state.mission.status == MissionStatus.READY:
            self.state.mission.status = MissionStatus.RUNNING

    def step(self, dt_s: float | None = None) -> list[SimulationEvent]:
        if dt_s is None:
            dt_s = self.config.simulation.timestep_s

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")

        if self.state.mission.status != MissionStatus.RUNNING:
            self.start()

        next_time = min(
            self.state.timestamp_s + dt_s,
            self.config.simulation.mission_duration_s,
        )

        ready_events = self.events.pop_ready(
            next_time
        )

        for event in ready_events:
            self._handle_event(event)

        # Medium-frequency network/decomposition update.
        if (
            self.state.timestamp_s
            >= self._next_network_update_s
        ):

            self._update_communication_graph(
                self.state.timestamp_s
            )

            self._next_network_update_s = (
                self.state.timestamp_s + 1.0
            )

        # Medium-frequency autonomy update.
        if (
            self.state.timestamp_s + 1e-9
            >= self._next_decision_update_s
        ):

            self.swarm.update(
                self.state,
                self.state.timestamp_s,
            )

            self._next_decision_update_s = (
                self.state.timestamp_s
                + self._decision_update_interval_s
            )

        actual_dt = (
            next_time
            - self.state.timestamp_s
        )

        if actual_dt > 0:

            # High-frequency physics/safety.
            self._update_kinematics(
                actual_dt
            )

            self._update_flight_state(
                actual_dt
            )

        self.state.timestamp_s = next_time
        self.state.mission.current_time_s = (
            next_time
        )

        # Refresh the network after physical motion when
        # the medium-frequency boundary has been reached.
        if (
            next_time + 1e-9
            >= self._next_network_update_s
        ):

            self._update_communication_graph(
                next_time
            )

            self._next_network_update_s = (
                next_time + 1.0
            )

        return ready_events


    def run(self, duration_s: float | None = None) -> SimulationState:
        if duration_s is None:
            duration_s = self.config.simulation.mission_duration_s

        if duration_s <= 0:
            raise ValueError("duration_s must be positive")

        target_time = min(
            self.state.timestamp_s + duration_s,
            self.config.simulation.mission_duration_s,
        )

        self.start()

        while (
            self.state.timestamp_s < target_time
            and self.state.mission.status == MissionStatus.RUNNING
        ):
            dt = min(
                self.config.simulation.timestep_s,
                target_time - self.state.timestamp_s,
            )
            self.step(dt)

        return self.state

    def _update_communication_graph(self, timestamp_s: float) -> None:
        """Rebuild the communication graph from current UAV positions."""

        positions = {
            uav_id: uav.position.copy()
            for uav_id, uav in self.state.uavs.items()
            if not uav.failure_state
            and uav.role != UAVRole.FAILED
        }

        graph = self.communication.build_graph(
            positions=positions,
            timestamp_s=timestamp_s,
            gcs_position=self.gcs_position,
        )

        self.state.network = graph

        # Reset runtime communication state.
        for uav in self.state.uavs.values():
            uav.neighbors.clear()
            uav.communication_quality = 0.0

        # Populate UAV-to-UAV neighbors and per-UAV communication quality.
        for (source_id, target_id), link in graph.links.items():
            if not link.availability:
                continue

            source_is_uav = source_id != GCS_NODE_ID
            target_is_uav = target_id != GCS_NODE_ID

            if source_is_uav and target_is_uav:
                self.state.uavs[source_id].neighbors.add(target_id)
                self.state.uavs[target_id].neighbors.add(source_id)

            if source_is_uav:
                self.state.uavs[source_id].communication_quality = max(
                    self.state.uavs[source_id].communication_quality,
                    link.link_quality,
                )

            if target_is_uav:
                self.state.uavs[target_id].communication_quality = max(
                    self.state.uavs[target_id].communication_quality,
                    link.link_quality,
                )

        self.network_analysis = NetworkGraphAnalyzer(graph).analyze()

    def _handle_event(self, event: SimulationEvent) -> None:
        if event.event_type == EventType.POI_SPAWN:
            if len(self.state.pois) >= self.config.scenario.poi_count:
                return

            poi = self.scenario_generator.create_poi(event.timestamp_s)
            self.state.pois[poi.poi_id] = poi

        elif event.event_type == EventType.MISSION_END:
            self.state.mission.status = MissionStatus.COMPLETED

    def _update_kinematics(self, dt_s: float) -> None:
        self.motion.advance(
            self.state,
            dt_s,
        )


    def _update_flight_state(self, dt_s: float) -> None:
        # Endurance is continuous flight endurance, not mission elapsed time.
        airborne_roles = {
            UAVRole.SURVEY,
            UAVRole.RELAY,
            UAVRole.TRANSIT,
            UAVRole.RECOVERY,
            UAVRole.RETURN_HOME,
        }

        for uav in self.state.uavs.values():
            if uav.failure_state:
                continue

            if uav.role not in airborne_roles:
                continue

            uav.flight_time_s += dt_s
            uav.battery.flight_time_s += dt_s

            uav.battery.soc = max(
                0.0,
                1.0
                - (
                    uav.flight_time_s
                    / self.config.uav.endurance_s
                ),
            )

            if (
                not uav.battery.rth_triggered
                and uav.battery.soc
                <= self.config.energy.rth_threshold
            ):
                uav.battery.rth_triggered = True
                self.events.push(
                    timestamp_s=self.state.timestamp_s,
                    event_type=EventType.RTH_TRIGGERED,
                    payload={"uav_id": uav.uav_id},
                )

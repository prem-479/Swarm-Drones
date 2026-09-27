
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np

from .allocation import CommunicationAwareAllocator
from .energy import EnergyManager
from .faults import RecoveryPlanner
from .motion import (
    AsymmetricCollisionAvoider,
    CommunicationPreservingPlanner,
    GeofenceGuard,
    GlobalPlanner,
)
from uavx.network import NetworkGraphAnalyzer
from uavx.core.config import UAVXConfig
from uavx.core.models import (
    Task,
    TaskStatus,
    UAVRole,
    UAVState,
)
from uavx.relay import RelaySelector
from uavx.roles import RoleManager


@dataclass
class DecisionStats:
    assignments: int = 0
    completed_tasks: int = 0
    expired_tasks: int = 0
    relay_reallocations: int = 0
    role_changes: int = 0
    recoveries: int = 0
    rth_events: int = 0


class SwarmDecisionEngine:
    """
    Runtime orchestration layer connecting mission, allocation,
    relay management, recovery, energy, motion and role state.
    """

    def __init__(self, config: UAVXConfig, gcs_position: np.ndarray) -> None:
        self.config = config
        self.gcs_position = np.asarray(gcs_position, dtype=float)

        self.allocator = CommunicationAwareAllocator()
        self.relay_selector = RelaySelector(
            max_range_m=config.communication.max_range_m,
            max_speed_mps=config.uav.max_speed_mps,
        )
        self.role_manager = RoleManager()
        self.energy_manager = EnergyManager(
            rth_threshold=config.energy.rth_threshold,
            reserve_fraction=config.energy.reserve_fraction,
        )
        self.recovery_planner = RecoveryPlanner()

        self.global_planner = GlobalPlanner()
        self.collision_avoider = AsymmetricCollisionAvoider(
            min_separation_m=config.uav.min_separation_m,
        )
        self.communication_motion = CommunicationPreservingPlanner(
            max_speed_mps=config.uav.max_speed_mps,
            comm_range_m=config.communication.max_range_m,
        )
        self.geofence = GeofenceGuard(
            lower=np.zeros(3, dtype=float),
            upper=np.array(
                [
                    config.environment.width_m,
                    config.environment.height_m,
                    config.environment.max_altitude_m,
                ],
                dtype=float,
            ),
        )

        self.stats = DecisionStats()

        # Role/reconfiguration stability policy.
        # These are controller heuristics, not organizer scoring formulas.
        self.min_role_duration_s = 10.0
        self.relay_change_cooldown_s = 10.0
        self.relay_reserve_fraction = 0.40

        self._last_task_replan_s = -np.inf
        self._last_relay_change_s = -np.inf
        self._stable_relays: set[int] = set()

    # --------------------------------------------------------
    # Main decision cycle
    # --------------------------------------------------------
    def update(self, state, timestamp_s: float) -> None:
        self._sync_tasks_from_pois(state, timestamp_s)
        self._process_completed_and_expired(state, timestamp_s)
        self._mark_failed_uavs(state)
        self._apply_energy_policy(state)

        if (
            timestamp_s - self._last_task_replan_s
            >= self.config.allocation.reevaluation_interval_s
        ):
            self._allocate_tasks(state)
            self._last_task_replan_s = timestamp_s

        relay_ids = self._select_relays(state)
        self._apply_roles(state, relay_ids, timestamp_s)

        self._set_targets(state)

        state.mission.completed_pois = sum(
            poi.status == TaskStatus.COMPLETED
            for poi in state.pois.values()
        )
        state.mission.missed_pois = sum(
            poi.status == TaskStatus.EXPIRED
            for poi in state.pois.values()
        )

    # --------------------------------------------------------
    # Task creation
    # --------------------------------------------------------
    def _sync_tasks_from_pois(self, state, timestamp_s: float) -> None:
        for poi_id, poi in state.pois.items():
            if poi_id in state.tasks:
                continue

            task = Task(
                task_id=poi_id,
                poi_id=poi_id,
                priority=float(poi.priority),
                created_time_s=float(timestamp_s),
                status=TaskStatus.PENDING,
                version=0,
            )

            state.tasks[task.task_id] = task
            state.mission.active_tasks.add(task.task_id)

    # --------------------------------------------------------
    # Task completion / discovery / reporting deadline
    # --------------------------------------------------------
    def _can_report(self, state, uav_id: int) -> bool:
        analysis = NetworkGraphAnalyzer(state.network).analyze(
            required_node_ids={uav_id},
        )
        return uav_id in analysis.gcs_reachable

    def _process_completed_and_expired(
        self,
        state,
        timestamp_s: float,
    ) -> None:
        for task in list(state.tasks.values()):
            if task.status not in {
                TaskStatus.ASSIGNED,
                TaskStatus.IN_PROGRESS,
                TaskStatus.PENDING,
            }:
                continue

            poi = state.pois.get(task.poi_id)
            if poi is None:
                continue

            if task.assigned_uav_id is not None:
                uav = state.uavs.get(task.assigned_uav_id)

                if uav is not None and not uav.failure_state:
                    distance = float(
                        np.linalg.norm(uav.position - poi.position)
                    )

                    if distance <= 2.0:
                        task.status = TaskStatus.IN_PROGRESS

                        # Reporting deadline starts at discovery.
                        if poi.discovery_time_s is None:
                            poi.discovery_time_s = timestamp_s
                            state.mission.discovered_pois.add(poi.poi_id)

                        # Under the official binary-range baseline,
                        # a GCS-reachable UAV can report immediately.
                        if self._can_report(state, uav.uav_id):
                            task.status = TaskStatus.COMPLETED
                            task.completion_time_s = timestamp_s
                            poi.status = TaskStatus.COMPLETED

                            state.mission.active_tasks.discard(task.task_id)

                            uav.assigned_task_id = None
                            uav.target = None
                            uav.mission_state = "IDLE"

                            self.stats.completed_tasks += 1
                            continue

            # A task cannot expire before the PoI has been discovered.
            if poi.discovery_time_s is None:
                continue

            deadline = (
                poi.discovery_time_s + poi.reporting_deadline_s
            )

            if timestamp_s > deadline:
                task.status = TaskStatus.EXPIRED
                poi.status = TaskStatus.EXPIRED
                state.mission.active_tasks.discard(task.task_id)

                if task.assigned_uav_id is not None:
                    assigned = state.uavs.get(task.assigned_uav_id)
                    if assigned is not None:
                        if assigned.assigned_task_id == task.task_id:
                            assigned.assigned_task_id = None
                            assigned.target = None
                            assigned.mission_state = "IDLE"

                task.assigned_uav_id = None
                self.stats.expired_tasks += 1

    # --------------------------------------------------------
    # Fault handling
    # --------------------------------------------------------
    def _mark_failed_uavs(self, state) -> None:
        for uav in state.uavs.values():
            if not uav.failure_state:
                continue

            task_id = uav.assigned_task_id
            if task_id is not None and task_id in state.tasks:
                state.tasks[task_id].assigned_uav_id = None
                if state.tasks[task_id].status not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.EXPIRED,
                }:
                    state.tasks[task_id].status = TaskStatus.PENDING

            uav.assigned_task_id = None
            uav.target = None
            uav.velocity[:] = 0.0
            uav.acceleration[:] = 0.0
            uav.role = UAVRole.FAILED

    def _recover_failed_assignments(self, state) -> None:
        candidates = {
            uav.uav_id: uav.position.copy()
            for uav in state.uavs.values()
            if not uav.failure_state
            and uav.role not in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }
        }

        battery = {
            uav.uav_id: float(uav.battery.soc)
            for uav in state.uavs.values()
        }
        margins = {
            uav.uav_id: float(uav.communication_quality)
            for uav in state.uavs.values()
        }
        burden = {
            uav.uav_id: float(
                1.0 if uav.assigned_task_id is not None else 0.0
            )
            for uav in state.uavs.values()
        }

        for task in state.tasks.values():
            if task.status not in {
                TaskStatus.PENDING,
                TaskStatus.ASSIGNED,
            }:
                continue

            if task.assigned_uav_id is not None:
                continue

            poi = state.pois.get(task.poi_id)
            if poi is None:
                continue

            action = self.recovery_planner.plan(
                failed_uav_id=-1,
                failed_position=poi.position,
                candidates=candidates,
                battery_fraction=battery,
                link_margin=margins,
                task_burden=burden,
            )

            if action.replacement_uav_id is None:
                continue

            replacement = state.uavs[action.replacement_uav_id]
            if replacement.assigned_task_id is not None:
                continue

            task.assigned_uav_id = replacement.uav_id
            task.status = TaskStatus.ASSIGNED
            replacement.assigned_task_id = task.task_id
            replacement.target = poi.position.copy()
            self.stats.recoveries += 1

    # --------------------------------------------------------
    # Allocation
    # --------------------------------------------------------
    def _allocate_tasks(self, state) -> None:
        active_relays = {
            uav.uav_id
            for uav in state.uavs.values()
            if uav.role == UAVRole.RELAY
            and not uav.failure_state
        }

        available = {
            uav.uav_id: uav.position.copy()
            for uav in state.uavs.values()
            if not uav.failure_state
            and uav.uav_id not in active_relays
            and uav.role not in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }
        }

        if not available:
            return

        total_fleet = len(state.uavs)

        # Preserve a dynamically sized reserve for relay deployment,
        # recovery and network reconfiguration.
        relay_reserve = max(
            2,
            int(np.ceil(total_fleet * self.relay_reserve_fraction)),
        )

        active_survey = sum(
            1
            for uav in state.uavs.values()
            if uav.assigned_task_id is not None
            and not uav.failure_state
            and uav.role not in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }
        )

        survey_capacity = max(1, total_fleet - relay_reserve)

        if active_survey >= survey_capacity:
            return

        remaining_capacity = survey_capacity - active_survey

        task_positions = [
            (task.task_id, state.pois[task.poi_id].position.copy())
            for task in sorted(
                state.tasks.values(),
                key=lambda item: (
                    -float(item.priority),
                    item.created_time_s,
                    item.task_id,
                ),
            )
            if task.status in {
                TaskStatus.PENDING,
                TaskStatus.ASSIGNED,
            }
            and task.assigned_uav_id is None
            and task.poi_id in state.pois
        ][:remaining_capacity]

        if not task_positions:
            return

        margins = {
            uav_id: float(
                1.0
                if state.uavs[uav_id].communication_quality > 0.0
                else -1.0
            )
            for uav_id in available
        }

        assignments = self.allocator.allocate(
            tasks=task_positions,
            uav_positions=available,
            active_relays=active_relays,
            max_speed_mps=self.config.uav.max_speed_mps,
            communication_margin=margins,
        )

        for task_id, uav_id in assignments.items():
            task = state.tasks.get(task_id)
            uav = state.uavs.get(uav_id)

            if task is None or uav is None:
                continue

            if task.assigned_uav_id is not None:
                continue

            task.assigned_uav_id = uav_id
            task.status = TaskStatus.ASSIGNED
            task.version += 1

            uav.assigned_task_id = task_id
            uav.mission_state = "ASSIGNED"

            poi = state.pois.get(task.poi_id)
            if poi is not None:
                uav.target = poi.position.copy()

            self.stats.assignments += 1

    # --------------------------------------------------------
    # Relay selection
    # --------------------------------------------------------
    def _select_relays(self, state) -> set[int]:
        if not self.config.relay.enabled:
            return set()

        survey_ids = {
            uav.uav_id
            for uav in state.uavs.values()
            if uav.assigned_task_id is not None
            and not uav.failure_state
        }

        if not survey_ids:
            return set()

        current_relays = {
            uav.uav_id
            for uav in state.uavs.values()
            if uav.role == UAVRole.RELAY
            and not uav.failure_state
        }

        analysis = NetworkGraphAnalyzer(state.network).analyze(
            required_node_ids=survey_ids,
        )

        disconnected = survey_ids - analysis.gcs_reachable

        # If the current relay set is protecting a connected swarm,
        # retain it. Do not oscillate merely because another candidate
        # gets a slightly different heuristic score.
        if not disconnected:
            self._stable_relays = set(current_relays)
            return set(current_relays)

        # Protect the existing relay configuration during its cooldown
        # unless it is no longer valid.
        if (
            current_relays
            and state.timestamp_s
            < self._last_relay_change_s + self.relay_change_cooldown_s
        ):
            self._stable_relays = set(current_relays)
            return set(current_relays)

        relay_budget = max(
            1,
            int(np.ceil(len(state.uavs) * self.relay_reserve_fraction)),
        )

        relay_budget = min(
            relay_budget,
            max(0, len(state.uavs) - len(survey_ids)),
        )

        if relay_budget <= 0:
            self._stable_relays = set()
            return set()

        selected = self.relay_selector.select_relays(
            uavs=state.uavs,
            graph=state.network,
            analysis=analysis,
            max_relays=relay_budget,
            backbone_nodes=[-1, *sorted(current_relays)],
            terminal_nodes=sorted(disconnected),
        )

        new_relays = set(current_relays).union(selected)

        # Never exceed the dynamically reserved relay capacity.
        if len(new_relays) > relay_budget:
            ranked = self.relay_selector.rank_candidates(
                uavs=state.uavs,
                graph=state.network,
                analysis=analysis,
                backbone_nodes=[-1, *sorted(current_relays)],
                terminal_nodes=sorted(disconnected),
            )

            ranked_ids = [item.uav_id for item in ranked]
            ordered = [
                relay_id for relay_id in current_relays
                if relay_id in new_relays
            ]

            for relay_id in ranked_ids:
                if relay_id not in ordered:
                    ordered.append(relay_id)

            new_relays = set(ordered[:relay_budget])

        if new_relays != current_relays:
            self._last_relay_change_s = state.timestamp_s
            self.stats.relay_reallocations += 1

        self._stable_relays = set(new_relays)
        return new_relays

    # --------------------------------------------------------
    # Role management
    # --------------------------------------------------------
    def _apply_roles(
        self,
        state,
        relay_ids: set[int],
        timestamp_s: float,
    ) -> None:
        for uav_id in sorted(state.uavs):
            uav = state.uavs[uav_id]
            role_state = state.roles.get(uav_id)

            if role_state is None:
                continue

            if uav.role in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }:
                continue

            # Minimum-role-duration guard.
            if (
                role_state.locked_until_s is not None
                and timestamp_s < role_state.locked_until_s
            ):
                continue

            if uav_id in relay_ids:
                desired_role = UAVRole.RELAY
                reason = "selected as communication relay"
            elif uav.assigned_task_id is not None:
                desired_role = UAVRole.SURVEY
                reason = "assigned active mission task"
            else:
                desired_role = UAVRole.RESERVE
                reason = "available without active relay or task"

            if desired_role == uav.role:
                continue

            transition = self.role_manager.transition(
                uav,
                desired_role,
                reason=reason,
            )

            role_state.previous_role = transition.previous_role
            role_state.current_role = transition.new_role
            role_state.changed_at_s = timestamp_s
            role_state.locked_until_s = (
                timestamp_s + self.min_role_duration_s
            )

            self.stats.role_changes += 1

    # --------------------------------------------------------
    # Energy policy
    # --------------------------------------------------------
    def _apply_energy_policy(self, state) -> None:
        for uav in state.uavs.values():
            if uav.failure_state:
                continue

            estimate = self.energy_manager.estimate(
                uav.battery.soc,
                self.config.uav.endurance_s,
            )

            if (
                estimate.should_rth
                and uav.role not in {
                    UAVRole.RETURN_HOME,
                    UAVRole.LANDING,
                    UAVRole.FAILED,
                }
            ):
                uav.battery.rth_triggered = True
                uav.role = UAVRole.RETURN_HOME
                uav.assigned_task_id = None
                uav.target = uav.home_position.copy()
                self.stats.rth_events += 1

    # --------------------------------------------------------
    # Target generation
    # --------------------------------------------------------
    def _set_targets(self, state) -> None:
        for uav in state.uavs.values():
            if uav.failure_state or uav.role == UAVRole.FAILED:
                uav.target = None
                continue

            if uav.role == UAVRole.RETURN_HOME:
                uav.target = uav.home_position.copy()
                continue

            if uav.role == UAVRole.RELAY:
                # Relay station keeping.
                uav.target = uav.position.copy()
                continue

            if uav.assigned_task_id is None:
                uav.target = None
                continue

            task = state.tasks.get(uav.assigned_task_id)
            if task is None:
                uav.assigned_task_id = None
                uav.target = None
                continue

            poi = state.pois.get(task.poi_id)
            if poi is not None and task.status not in {
                TaskStatus.COMPLETED,
                TaskStatus.EXPIRED,
            }:
                uav.target = poi.position.copy()

    # --------------------------------------------------------
    # Motion / safety
    # --------------------------------------------------------
    def compute_velocity(self, state, uav_id: int) -> np.ndarray:
        uav = state.uavs[uav_id]

        if uav.target is None:
            return np.zeros(3, dtype=float)

        preferred = self.global_planner.velocity_toward(
            current=uav.position,
            target=uav.target,
        )

        neighbor_positions = {
            other_id: other.position.copy()
            for other_id, other in state.uavs.items()
            if other_id != uav_id
            and not other.failure_state
        }

        velocity = self.collision_avoider.filter_velocity(
            uav_position=uav.position,
            desired_velocity=preferred,
            neighbors=neighbor_positions,
            uav_role=uav.role.value,
        )

        anchors = [self.gcs_position]
        anchors.extend(
            other.position.copy()
            for other in state.uavs.values()
            if other.role == UAVRole.RELAY
            and other.uav_id != uav_id
            and not other.failure_state
        )

        candidate = self.communication_motion.choose(
            current=uav.position,
            preferred=velocity,
            anchor_positions=anchors,
        )

        velocity = candidate.velocity

        if uav.role == UAVRole.RELAY and not candidate.preserves_connectivity:
            velocity = np.zeros(3, dtype=float)

        velocity = self.geofence.clamp_velocity(
            position=uav.position,
            velocity=velocity,
        )

        return velocity

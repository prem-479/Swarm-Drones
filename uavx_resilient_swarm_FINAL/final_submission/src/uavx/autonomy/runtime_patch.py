from __future__ import annotations

import numpy as np

from uavx.core.models import UAVRole, TaskStatus
from uavx.autonomy.backbone import PredictiveRelayBackbone
from uavx.autonomy.controller import SwarmDecisionEngine


_INSTALLED = False


def install() -> None:
    global _INSTALLED

    if _INSTALLED:
        return

    original_select = SwarmDecisionEngine._select_relays
    original_set_targets = SwarmDecisionEngine._set_targets

    def predictive_select(self, state) -> set[int]:
        if not self.config.relay.enabled:
            return set()

        targets = []

        for task in state.tasks.values():
            if task.assigned_uav_id is None:
                continue

            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.EXPIRED,
                TaskStatus.CANCELLED,
            }:
                continue

            poi = state.pois.get(task.poi_id)
            if poi is not None:
                targets.append(poi.position.copy())

        if not targets:
            self._relay_waypoints = {}
            return set()

        # Protect active survey UAVs.
        survey_ids = {
            uav.uav_id
            for uav in state.uavs.values()
            if uav.assigned_task_id is not None
            and not uav.failure_state
        }

        # Use the currently farthest active mission target.
        target = max(
            targets,
            key=lambda p: float(np.linalg.norm(p[:2])),
        )

        candidates = [
            uav
            for uav in state.uavs.values()
            if uav.uav_id not in survey_ids
            and not uav.failure_state
            and uav.role not in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }
        ]

        # Stable ordering prevents relay oscillation.
        candidates.sort(key=lambda u: u.uav_id)

        planner = PredictiveRelayBackbone(
            max_range_m=self.config.communication.max_range_m,
            safety_margin_m=10.0,
        )

        waypoints = planner.plan(target, candidates)

        self._relay_waypoints = {
            waypoint.uav_id: waypoint.position.copy()
            for waypoint in waypoints
        }

        return set(self._relay_waypoints)

    def patched_set_targets(self, state) -> None:
        original_set_targets(self, state)

        relay_waypoints = getattr(self, "_relay_waypoints", {})

        for uav_id, position in relay_waypoints.items():
            uav = state.uavs.get(uav_id)

            if uav is None or uav.failure_state:
                continue

            if uav.role == UAVRole.RELAY:
                uav.target = np.asarray(position, dtype=float).copy()

    SwarmDecisionEngine._select_relays = predictive_select
    SwarmDecisionEngine._set_targets = patched_set_targets

    _INSTALLED = True

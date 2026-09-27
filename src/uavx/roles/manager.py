from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from uavx.core.models import UAVRole, UAVState


@dataclass(frozen=True)
class RoleTransition:
    uav_id: int
    previous_role: UAVRole
    new_role: UAVRole
    reason: str


class RoleManager:
    """
    Deterministic UAV role state machine.

    This phase only manages role transitions.
    Failure recovery, battery handover, motion planning, and
    task allocation remain separate phases.
    """

    _ALLOWED_TRANSITIONS: dict[UAVRole, frozenset[UAVRole]] = {
        UAVRole.IDLE: frozenset({
            UAVRole.RESERVE,
            UAVRole.SURVEY,
            UAVRole.RELAY,
            UAVRole.RETURN_HOME,
        }),
        UAVRole.RESERVE: frozenset({
            UAVRole.SURVEY,
            UAVRole.RELAY,
            UAVRole.TRANSIT,
            UAVRole.RECOVERY,
            UAVRole.RETURN_HOME,
            UAVRole.LANDING,
            UAVRole.FAILED,
        }),
        UAVRole.SURVEY: frozenset({
            UAVRole.RELAY,
            UAVRole.TRANSIT,
            UAVRole.RECOVERY,
            UAVRole.RESERVE,
            UAVRole.RETURN_HOME,
            UAVRole.LANDING,
            UAVRole.FAILED,
        }),
        UAVRole.RELAY: frozenset({
            UAVRole.SURVEY,
            UAVRole.TRANSIT,
            UAVRole.RECOVERY,
            UAVRole.RESERVE,
            UAVRole.RETURN_HOME,
            UAVRole.LANDING,
            UAVRole.FAILED,
        }),
        UAVRole.TRANSIT: frozenset({
            UAVRole.SURVEY,
            UAVRole.RELAY,
            UAVRole.RECOVERY,
            UAVRole.RESERVE,
            UAVRole.RETURN_HOME,
            UAVRole.LANDING,
            UAVRole.FAILED,
        }),
        UAVRole.RECOVERY: frozenset({
            UAVRole.RETURN_HOME,
            UAVRole.LANDING,
            UAVRole.FAILED,
            UAVRole.RESERVE,
        }),
        UAVRole.RETURN_HOME: frozenset({
            UAVRole.LANDING,
            UAVRole.FAILED,
        }),
        UAVRole.LANDING: frozenset({
            UAVRole.FAILED,
            UAVRole.IDLE,
            UAVRole.RESERVE,
        }),
        UAVRole.FAILED: frozenset(),
    }

    def can_transition(
        self,
        previous_role: UAVRole,
        new_role: UAVRole,
    ) -> bool:
        if previous_role == new_role:
            return True
        return new_role in self._ALLOWED_TRANSITIONS.get(previous_role, frozenset())

    def transition(
        self,
        uav: UAVState,
        new_role: UAVRole,
        *,
        reason: str,
    ) -> RoleTransition:
        previous_role = uav.role

        if not self.can_transition(previous_role, new_role):
            raise ValueError(
                f"Invalid role transition for UAV {uav.uav_id}: "
                f"{previous_role.value} -> {new_role.value}"
            )

        uav.role = new_role

        return RoleTransition(
            uav_id=uav.uav_id,
            previous_role=previous_role,
            new_role=new_role,
            reason=reason,
        )

    def evaluate(
        self,
        uavs: Mapping[int, UAVState],
        *,
        relay_ids: set[int] | None = None,
    ) -> list[RoleTransition]:
        """
        Apply deterministic role decisions for the current state.

        Priority:
        1. Failed / return-home / landing states are preserved.
        2. Explicit relay assignment becomes RELAY.
        3. UAVs with assigned active tasks become SURVEY.
        4. Unassigned available UAVs remain RESERVE.
        """
        relay_ids = relay_ids or set()
        transitions: list[RoleTransition] = []

        for uav_id in sorted(uavs):
            uav = uavs[uav_id]

            if uav.role in {
                UAVRole.FAILED,
                UAVRole.RETURN_HOME,
                UAVRole.LANDING,
            }:
                continue

            if uav_id in relay_ids:
                desired_role = UAVRole.RELAY
                reason = "selected as communication relay"
            elif uav.assigned_task_id is not None:
                desired_role = UAVRole.SURVEY
                reason = "assigned active mission task"
            else:
                desired_role = UAVRole.RESERVE
                reason = "available without active relay or task assignment"

            if desired_role == uav.role:
                continue

            transitions.append(
                self.transition(
                    uav,
                    desired_role,
                    reason=reason,
                )
            )

        return transitions

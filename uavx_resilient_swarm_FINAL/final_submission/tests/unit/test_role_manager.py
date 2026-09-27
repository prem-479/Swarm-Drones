from __future__ import annotations

import numpy as np
import pytest

from uavx.core.models import BatteryState, UAVRole, UAVState
from uavx.roles import RoleManager


def make_uav(
    uav_id: int,
    role: UAVRole = UAVRole.RESERVE,
    task_id: int | None = None,
) -> UAVState:
    return UAVState(
        uav_id=uav_id,
        position=np.array([0.0, 0.0, 20.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        role=role,
        battery=BatteryState(
            soc=1.0,
            flight_time_s=0.0,
            energy_consumed=0.0,
            rth_triggered=False,
        ),
        assigned_task_id=task_id,
    )


def test_valid_transition_is_allowed() -> None:
    manager = RoleManager()

    assert manager.can_transition(
        UAVRole.RESERVE,
        UAVRole.RELAY,
    )


def test_invalid_transition_is_rejected() -> None:
    manager = RoleManager()

    assert not manager.can_transition(
        UAVRole.FAILED,
        UAVRole.SURVEY,
    )


def test_transition_updates_uav_role() -> None:
    manager = RoleManager()
    uav = make_uav(0)

    transition = manager.transition(
        uav,
        UAVRole.RELAY,
        reason="relay assignment",
    )

    assert uav.role == UAVRole.RELAY
    assert transition.uav_id == 0
    assert transition.previous_role == UAVRole.RESERVE
    assert transition.new_role == UAVRole.RELAY
    assert transition.reason == "relay assignment"


def test_invalid_transition_raises() -> None:
    manager = RoleManager()
    uav = make_uav(0, UAVRole.FAILED)

    with pytest.raises(ValueError):
        manager.transition(
            uav,
            UAVRole.RELAY,
            reason="invalid recovery",
        )


def test_evaluate_assigns_relay_role() -> None:
    manager = RoleManager()

    uavs = {
        0: make_uav(0),
        1: make_uav(1, UAVRole.SURVEY, task_id=10),
    }

    transitions = manager.evaluate(
        uavs,
        relay_ids={0},
    )

    assert uavs[0].role == UAVRole.RELAY
    assert len(transitions) == 1
    assert transitions[0].uav_id == 0


def test_evaluate_assigns_survey_role_for_task() -> None:
    manager = RoleManager()

    uavs = {
        0: make_uav(0, UAVRole.RESERVE, task_id=5),
    }

    transitions = manager.evaluate(uavs)

    assert uavs[0].role == UAVRole.SURVEY
    assert len(transitions) == 1
    assert transitions[0].new_role == UAVRole.SURVEY


def test_unassigned_available_uav_becomes_reserve() -> None:
    manager = RoleManager()

    uavs = {
        0: make_uav(0, UAVRole.SURVEY),
    }

    transitions = manager.evaluate(uavs)

    assert uavs[0].role == UAVRole.RESERVE
    assert len(transitions) == 1


def test_terminal_roles_are_preserved() -> None:
    manager = RoleManager()

    uavs = {
        0: make_uav(0, UAVRole.FAILED),
        1: make_uav(1, UAVRole.RETURN_HOME),
        2: make_uav(2, UAVRole.LANDING),
    }

    transitions = manager.evaluate(
        uavs,
        relay_ids={0, 1, 2},
    )

    assert transitions == []
    assert uavs[0].role == UAVRole.FAILED
    assert uavs[1].role == UAVRole.RETURN_HOME
    assert uavs[2].role == UAVRole.LANDING


def test_evaluation_is_deterministic() -> None:
    manager = RoleManager()

    uavs = {
        2: make_uav(2, UAVRole.RESERVE),
        0: make_uav(0, UAVRole.RESERVE, task_id=1),
        1: make_uav(1, UAVRole.RESERVE),
    }

    transitions = manager.evaluate(
        uavs,
        relay_ids={1},
    )

    assert [t.uav_id for t in transitions] == [0, 1]
    assert uavs[0].role == UAVRole.SURVEY
    assert uavs[1].role == UAVRole.RELAY
    assert uavs[2].role == UAVRole.RESERVE

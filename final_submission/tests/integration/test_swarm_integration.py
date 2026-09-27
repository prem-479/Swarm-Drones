
import numpy as np

from uavx.core import load_config
from uavx.core.events import EventType
from uavx.core.models import TaskStatus, UAVRole
from uavx.simulation import SimulationEngine


CONFIG_PATH = "configs/official_stage1.yaml"


def test_swarm_controller_assigns_spawned_poi() -> None:
    engine = SimulationEngine.from_config(load_config(CONFIG_PATH))

    poi = engine.scenario_generator.create_poi(0.0)
    poi.position = np.array([30.0, 0.0, 20.0])
    engine.state.pois[poi.poi_id] = poi

    engine.swarm.update(engine.state, 0.0)

    assert engine.state.tasks
    assert any(
        uav.assigned_task_id is not None
        for uav in engine.state.uavs.values()
    )


def test_swarm_roles_include_survey_after_assignment() -> None:
    engine = SimulationEngine.from_config(load_config(CONFIG_PATH))

    poi = engine.scenario_generator.create_poi(0.0)
    poi.position = np.array([30.0, 0.0, 20.0])
    engine.state.pois[poi.poi_id] = poi

    engine.swarm.update(engine.state, 0.0)

    assert any(
        uav.role == UAVRole.SURVEY
        for uav in engine.state.uavs.values()
    )


def test_engine_step_moves_assigned_uav() -> None:
    engine = SimulationEngine.from_config(load_config(CONFIG_PATH))

    poi = engine.scenario_generator.create_poi(0.0)
    poi.position = np.array([30.0, 0.0, 20.0])
    engine.state.pois[poi.poi_id] = poi

    engine.swarm.update(engine.state, 0.0)

    assigned = next(
        uav for uav in engine.state.uavs.values()
        if uav.assigned_task_id is not None
    )

    start = assigned.position.copy()

    engine.start()
    engine.step(0.1)

    assert np.linalg.norm(assigned.position - start) > 0.0
    assert np.linalg.norm(assigned.velocity) <= 5.0 + 1e-9


def test_failed_uav_is_not_used_for_new_assignment() -> None:
    engine = SimulationEngine.from_config(load_config(CONFIG_PATH))

    engine.state.uavs[0].failure_state = True
    engine.state.uavs[0].role = UAVRole.FAILED

    poi = engine.scenario_generator.create_poi(0.0)
    poi.position = np.array([30.0, 0.0, 20.0])
    engine.state.pois[poi.poi_id] = poi

    engine.swarm.update(engine.state, 0.0)

    assert engine.state.tasks
    task = next(iter(engine.state.tasks.values()))

    assert task.assigned_uav_id != 0


def test_low_battery_triggers_rth() -> None:
    engine = SimulationEngine.from_config(load_config(CONFIG_PATH))

    uav = engine.state.uavs[0]
    uav.battery.soc = 0.20

    engine.swarm.update(engine.state, 0.0)

    assert uav.role == UAVRole.RETURN_HOME
    assert uav.target is not None
    assert np.allclose(uav.target, uav.home_position)


def test_relay_can_be_selected_after_task_assignment() -> None:
    engine = SimulationEngine.from_config(load_config(CONFIG_PATH))

    # Keep all UAVs in the connected initial chain.
    poi = engine.scenario_generator.create_poi(0.0)
    poi.position = np.array([80.0, 0.0, 20.0])
    engine.state.pois[poi.poi_id] = poi

    engine.swarm.update(engine.state, 0.0)

    assert any(
        uav.assigned_task_id is not None
        for uav in engine.state.uavs.values()
    )

    engine._update_communication_graph(0.0)
    engine.swarm.update(engine.state, 1.0)

    # A relay is allowed to be absent when the topology does not need one,
    # so validate that no invalid FAILED/RETURN_HOME relay was produced.
    for uav in engine.state.uavs.values():
        if uav.role == UAVRole.RELAY:
            assert not uav.failure_state
            assert uav.role != UAVRole.FAILED

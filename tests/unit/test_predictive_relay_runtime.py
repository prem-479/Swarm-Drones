import numpy as np

from uavx.core.config import load_config
from uavx.core.models import PoI, Task, TaskStatus
from uavx.simulation.engine import SimulationEngine


CONFIG = "configs/stage1.yaml"


def test_far_target_gets_predictive_relay_waypoints():
    engine = SimulationEngine.from_config(load_config(CONFIG))

    poi = PoI(
        poi_id=100,
        position=np.array([950.0, 0.0, 20.0]),
        priority=5.0,
        spawn_time_s=0.0,
    )

    task = Task(
        task_id=100,
        poi_id=100,
        priority=5.0,
        created_time_s=0.0,
        status=TaskStatus.ASSIGNED,
        assigned_uav_id=0,
    )

    engine.state.pois[poi.poi_id] = poi
    engine.state.tasks[task.task_id] = task
    engine.state.uavs[0].assigned_task_id = task.task_id

    engine.swarm.update(engine.state, 0.0)

    waypoints = getattr(engine.swarm, "_relay_waypoints", {})

    assert len(waypoints) >= 8
    assert all(
        0.0 <= p[0] <= 1000.0 and
        0.0 <= p[1] <= 1000.0
        for p in waypoints.values()
    )

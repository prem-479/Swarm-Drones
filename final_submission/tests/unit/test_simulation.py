import numpy as np

from uavx.core.config import load_config
from uavx.core.events import EventQueue, EventType
from uavx.simulation import SimulationEngine


CONFIG = "configs/official_stage1.yaml"


def test_event_queue_orders_events():
    queue = EventQueue()

    queue.push(5.0, EventType.MISSION_END)
    queue.push(1.0, EventType.POI_SPAWN)
    queue.push(3.0, EventType.POI_DISCOVERED)

    events = queue.pop_ready(3.0)

    assert [event.timestamp_s for event in events] == [1.0, 3.0]


def test_simulation_initializes_uavs():
    config = load_config(CONFIG)
    engine = SimulationEngine.from_config(config)

    assert len(engine.state.uavs) == config.uav.count
    assert engine.state.timestamp_s == 0.0


def test_simulation_advances():
    config = load_config(CONFIG)
    engine = SimulationEngine.from_config(config)

    engine.start()
    engine.step(1.0)

    assert engine.state.timestamp_s == 1.0
    assert engine.state.mission.current_time_s == 1.0


def test_uav_respects_speed_limit():
    config = load_config(CONFIG)
    engine = SimulationEngine.from_config(config)

    uav = engine.state.uavs[0]
    uav.target = np.array([1000.0, 0.0, 20.0])

    engine.start()
    engine.step(1.0)

    speed = float(np.linalg.norm(uav.velocity))

    assert speed <= config.uav.max_speed_mps + 1e-9


def test_poi_generation_is_seeded():
    config = load_config(CONFIG)

    engine_a = SimulationEngine.from_config(config)
    engine_b = SimulationEngine.from_config(config)

    engine_a.start()
    engine_b.start()

    for _ in range(20):
        engine_a.step(0.1)
        engine_b.step(0.1)

    pois_a = list(engine_a.state.pois.values())
    pois_b = list(engine_b.state.pois.values())

    assert len(pois_a) == len(pois_b)

    for a, b in zip(pois_a, pois_b):
        assert np.allclose(a.position, b.position)
        assert a.priority == b.priority
        assert a.spawn_time_s == b.spawn_time_s

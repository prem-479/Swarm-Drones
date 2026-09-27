from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    timestep_s: float
    mission_duration_s: float


@dataclass(frozen=True)
class EnvironmentConfig:
    width_m: float
    height_m: float
    max_altitude_m: float


@dataclass(frozen=True)
class UAVConfig:
    count: int
    max_speed_mps: float
    max_acceleration_mps2: float
    endurance_s: float
    min_separation_m: float


@dataclass(frozen=True)
class CommunicationConfig:
    max_range_m: float
    model: str
    packet_delivery_probability: float
    latency_ms: float


@dataclass(frozen=True)
class MissionConfig:
    poi_reporting_deadline_s: float


@dataclass(frozen=True)
class EnergyConfig:
    rth_threshold: float
    reserve_fraction: float


@dataclass(frozen=True)
class SafetyConfig:
    enforce_geofence: bool
    enforce_min_separation: bool


@dataclass(frozen=True)
class AllocationConfig:
    method: str
    reevaluation_interval_s: float


@dataclass(frozen=True)
class RelayConfig:
    enabled: bool
    redundancy: bool


@dataclass(frozen=True)
class ScenarioConfig:
    poi_count: int
    poi_dynamic: bool


@dataclass(frozen=True)
class UAVXConfig:
    simulation: SimulationConfig
    environment: EnvironmentConfig
    uav: UAVConfig
    communication: CommunicationConfig
    mission: MissionConfig
    energy: EnergyConfig
    safety: SafetyConfig
    allocation: AllocationConfig
    relay: RelayConfig
    scenario: ScenarioConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> UAVXConfig:
    """Load and validate a UAV-X YAML configuration."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    simulation = _require(raw, "simulation")
    environment = _require(raw, "environment")
    uav = _require(raw, "uav")
    communication = _require(raw, "communication")
    mission = _require(raw, "mission")
    energy = _require(raw, "energy")
    safety = _require(raw, "safety")
    allocation = _require(raw, "allocation")
    relay = _require(raw, "relay")

    # Scenario defaults are provided for compatibility with earlier configs.
    scenario = raw.get(
        "scenario",
        {
            "poi_count": 10,
            "poi_dynamic": True,
        },
    )

    cfg = UAVXConfig(
        simulation=SimulationConfig(
            seed=int(_require(simulation, "seed")),
            timestep_s=float(_require(simulation, "timestep_s")),
            mission_duration_s=float(_require(simulation, "mission_duration_s")),
        ),
        environment=EnvironmentConfig(
            width_m=float(_require(environment, "width_m")),
            height_m=float(_require(environment, "height_m")),
            max_altitude_m=float(_require(environment, "max_altitude_m")),
        ),
        uav=UAVConfig(
            count=int(_require(uav, "count")),
            max_speed_mps=float(_require(uav, "max_speed_mps")),
            max_acceleration_mps2=float(_require(uav, "max_acceleration_mps2")),
            endurance_s=float(_require(uav, "endurance_s")),
            min_separation_m=float(_require(uav, "min_separation_m")),
        ),
        communication=CommunicationConfig(
            max_range_m=float(_require(communication, "max_range_m")),
            model=str(_require(communication, "model")),
            packet_delivery_probability=float(
                _require(communication, "packet_delivery_probability")
            ),
            latency_ms=float(_require(communication, "latency_ms")),
        ),
        mission=MissionConfig(
            poi_reporting_deadline_s=float(
                _require(mission, "poi_reporting_deadline_s")
            )
        ),
        energy=EnergyConfig(
            rth_threshold=float(_require(energy, "rth_threshold")),
            reserve_fraction=float(_require(energy, "reserve_fraction")),
        ),
        safety=SafetyConfig(
            enforce_geofence=bool(_require(safety, "enforce_geofence")),
            enforce_min_separation=bool(
                _require(safety, "enforce_min_separation")
            ),
        ),
        allocation=AllocationConfig(
            method=str(_require(allocation, "method")),
            reevaluation_interval_s=float(
                _require(allocation, "reevaluation_interval_s")
            ),
        ),
        relay=RelayConfig(
            enabled=bool(_require(relay, "enabled")),
            redundancy=bool(_require(relay, "redundancy")),
        ),
        scenario=ScenarioConfig(
            poi_count=int(_require(scenario, "poi_count")),
            poi_dynamic=bool(_require(scenario, "poi_dynamic")),
        ),
    )

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: UAVXConfig) -> None:
    if cfg.simulation.timestep_s <= 0:
        raise ValueError("simulation.timestep_s must be > 0")

    if cfg.simulation.mission_duration_s <= 0:
        raise ValueError("mission duration must be > 0")

    if cfg.environment.width_m <= 0 or cfg.environment.height_m <= 0:
        raise ValueError("environment dimensions must be > 0")

    if cfg.environment.max_altitude_m <= 0:
        raise ValueError("maximum altitude must be > 0")

    if cfg.uav.count <= 0:
        raise ValueError("uav.count must be > 0")

    if cfg.uav.max_speed_mps <= 0:
        raise ValueError("maximum UAV speed must be > 0")

    if cfg.uav.endurance_s <= 0:
        raise ValueError("UAV endurance must be > 0")

    if cfg.communication.max_range_m <= 0:
        raise ValueError("communication range must be > 0")

    if not 0.0 <= cfg.communication.packet_delivery_probability <= 1.0:
        raise ValueError("packet delivery probability must be in [0, 1]")

    if not 0.0 <= cfg.energy.rth_threshold <= 1.0:
        raise ValueError("rth_threshold must be in [0, 1]")

    if not 0.0 <= cfg.energy.reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in [0, 1)")

    if cfg.scenario.poi_count < 0:
        raise ValueError("poi_count cannot be negative")

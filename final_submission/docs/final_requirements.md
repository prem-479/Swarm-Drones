# UAV-X Resilient BVLOS Swarm
# Final Requirements Audit
# PUSHPAK Grand Challenge 2026

## 1. Authority Hierarchy

For implementation purposes, requirements are interpreted in this order:

1. Official UAV-X Problem Statement
2. Official written clarifications issued by the organizers
3. PUSHPAK Grand Challenge Terms and Conditions
4. Mathematical / technical inference from official requirements
5. Research-backed recommendations
6. Engineering assumptions

Research material must not override an explicit official requirement.

---

## 2. Official Requirements

| Parameter | Value | Unit | Category | Source | How Implemented | Config Key |
|---|---:|---|---|---|---|---|
| Operational area width | 1000 | m | A | Official UAV-X Problem Statement | Hard geofence | `environment.width_m` |
| Operational area height | 1000 | m | A | Official UAV-X Problem Statement | Hard geofence | `environment.height_m` |
| Maximum communication range | 100 | m | A | Official UAV-X Problem Statement | Communication graph edge boundary | `communication.max_range_m` |
| UAV flight endurance | 20 | min / 1200 s | A | Official UAV-X Problem Statement | Battery / endurance model | `uav.endurance_s` |
| Mission duration | 45 | min / 2700 s | A | Official UAV-X Problem Statement | Simulation termination | `simulation.mission_duration_s` |
| Minimum inter-UAV separation | 20 | m | A | Official UAV-X Problem Statement | Collision safety constraint | `safety.min_separation_m` |
| PoI reporting deadline | 10 | s | A | Official UAV-X Problem Statement | Reporting deadline / latency metric | `mission.report_deadline_s` |
| Maximum UAV speed | 5 | m/s | A | Official Stage-1 scenario | Kinematic speed constraint | `uav.max_speed_mps` |
| Maximum altitude | 100 | m | A | Official Stage-1 scenario | Altitude/geofence constraint | `environment.max_altitude_m` |
| PoI count | 10 | count | A | Official Stage-1 implementation specification | Scenario generation limit | `scenario.poi_count` |
| Dynamic PoI spawning | enabled / stochastic | boolean | A | Official Stage-1 implementation specification | Event-driven PoI generator | `scenario.poi_dynamic` |

---

## 3. Mission-Level Requirements

| Requirement | Category | Implementation |
|---|---|---|
| Survey assigned disaster locations | A | Mission manager + task execution |
| Service dynamically appearing PoIs | A | Event-driven task generation |
| Maintain end-to-end GCS communication | A | Dynamic network graph |
| Multi-hop aerial communication | A | Graph routing / relay layer |
| Dynamically assign relay UAVs | A | Relay manager |
| Recover from UAV failures | A | Failure detection + recovery |
| Recover from communication degradation | A | Topology monitoring + reconfiguration |
| Handle return-to-home / energy constraints | A | Energy manager + RTH |
| Prioritize newly emerging high-priority tasks | A | Priority-aware dynamic allocation |
| Avoid collisions | A | Safety / collision layer |
| Respect geofence | A | Environment safety layer |
| Return safely | A | RTH + landing state machine |
| Complete within mission window | A | Mission termination / metrics |

---

## 4. Mathematical / Technical Inferences

| Parameter / Result | Value | Unit | Category | Reason | Implementation |
|---|---:|---|---|---|---|
| Maximum geometric distance from GCS to area corner | 1414.21 | m | B | sqrt(1000² + 1000²) | Analysis only |
| Minimum theoretical hops to furthest corner | 15 | hops | B | ceil(1414.21 / 100) | Network analysis |
| Intermediate relay count for a single 15-hop chain | 14 | UAVs | B | Single-chain topology | Relay planner baseline |
| Mission/endurance ratio | 2.25 | ratio | B | 2700 / 1200 | Energy scheduling analysis |

### Important inference limitation

The 15-hop result is a geometric lower bound under idealized connectivity.
It does NOT mean that exactly 14 relay UAVs are always required.

Actual relay count depends on:

- UAV positions
- network topology
- redundancy
- survey UAV locations
- communication model
- obstacle / safety constraints
- mission strategy

The system must therefore compute topology dynamically.

---

## 5. Research-Backed Recommendations

| Parameter / Mechanism | Recommendation | Category | Implementation |
|---|---|---|---|
| Communication model | Distance-dependent / probabilistic model | C | Configurable communication model |
| Advanced communication model | Log-distance/path-loss + stochastic packet loss | C | Optional model |
| Network analysis | Connected components | C | Network engine |
| Network analysis | Shortest paths | C | Network engine |
| Network analysis | Articulation point detection | C | Network engine |
| Network analysis | Bridge detection | C | Network engine |
| Network analysis | Path redundancy | C | Network engine |
| Relay selection | Connectivity contribution | C | Relay scoring |
| Relay selection | Link margin | C | Relay scoring |
| Relay selection | Centrality | C | Relay scoring |
| Relay selection | Articulation risk | C | Relay scoring |
| Relay selection | Battery state | C | Relay scoring |
| Relay selection | Task burden | C | Relay scoring |
| Relay selection | Future mobility | C | Relay scoring |
| Relay strategy | Primary + backup paths | C | Redundancy manager |
| Task allocation | Communication-aware utility | C | Allocation engine |
| Task allocation | Urgency-aware allocation | C | Allocation engine |
| Task allocation | Network disruption cost | C | Allocation engine |
| State synchronization | Versioned state | C | Distributed state model |
| Collision avoidance | VR-ORCA-like asymmetric behavior | C | Safety layer |
| Motion planning | Communication-preserving trajectory selection | C | Motion planner |
| Scenario testing | Monte Carlo with multiple seeds | C | Experiment framework |

---

## 6. Engineering Assumptions

These parameters are NOT official unless later confirmed by organizers.

| Parameter | Current Value | Unit | Category | Reason | Config Key |
|---|---:|---|---|---|---|
| Maximum acceleration | 2.0 | m/s² | D | Research / engineering assumption | `uav.max_acceleration_mps2` |
| Initial fleet size | 10 | UAVs | D | Initial simulation choice | `uav.count` |
| Battery RTH threshold | 25 | % | D | Initial controller assumption | `energy.rth_threshold` |
| Energy reserve fraction | 15 | % | D | Initial safety assumption | `energy.reserve_fraction` |
| Simulation timestep | 0.1 | s | D | Numerical integration choice | `simulation.timestep_s` |
| Default random seed | 42 | - | D | Reproducibility convention | `simulation.seed` |
| Initial communication model | binary range | - | D | Simple baseline | `communication.model` |
| Default packet delivery probability | 1.0 | probability | D | Ideal baseline | `communication.packet_delivery_probability` |
| Default latency | 0 | ms | D | Ideal baseline | `communication.latency_ms` |
| Allocation reevaluation interval | 1.0 | s | D | Controller scheduling choice | `allocation.reevaluation_interval_s` |

---

## 7. Contradictions and Resolutions

### C-01: Maximum UAV speed

Research material contains a 10 m/s assumption.

Official Stage-1 implementation baseline specifies:

5 m/s

Resolution:

USE 5 m/s FOR THE OFFICIAL BASELINE.

10 m/s must NOT be silently used.

Alternative speed values may be used only in explicitly named experimental configurations.

---

### C-02: Fleet size

Research material discusses fleet sizes such as 10–30 UAVs.

This is treated as an engineering/research parameter rather than an official fixed fleet size unless the organizers provide an explicit fleet specification.

Resolution:

Fleet size remains configurable.

---

### C-03: Relay count

Research analysis describes a 15-hop worst-case chain requiring 14 intermediate relays.

Resolution:

This is a mathematical worst-case topology result, NOT a fixed requirement.

The relay manager must dynamically determine the required topology.

---

### C-04: Communication model

A 100 m range is an official hard communication boundary.

A binary range model with perfect communication is only a baseline simulation assumption.

Resolution:

Keep the hard maximum range at 100 m while allowing realistic packet loss, latency and link quality models in separate configurations.

---

### C-05: Endurance-derived fleet rotation

Some research calculations derive additional rotation requirements from transit time using 10 m/s.

Because the official baseline speed is 5 m/s, those derived numerical values must not be treated as official.

Resolution:

Recompute energy / transit-derived quantities using the active configuration.

Do not hard-code research-derived fleet multipliers.

---

## 8. Parameters That Must Remain Configurable

The following must never be hard-coded into algorithm logic:

- Fleet size
- Maximum acceleration
- Random seed
- PoI positions
- PoI spawn times
- PoI priorities
- Service duration
- Communication model
- Packet delivery probability
- Communication latency
- Battery thresholds
- Energy reserve
- Task allocation weights
- Relay scoring weights
- Motion-planning horizon
- Failure timing
- Link outage timing
- Number of failures

---

## 9. Source-of-Truth Rule

All official competition parameters must be represented once in configuration.

Algorithm modules must consume configuration values.

No subsystem may silently replace an official value with a research assumption.

Every derived quantity must identify:

- source parameter
- mathematical derivation
- assumptions
- resulting value

---

## 10. Current Implementation Baseline

The first reproducible baseline is:

- Area: 1000 x 1000 m
- Maximum altitude: 100 m
- Maximum UAV speed: 5 m/s
- UAV endurance: 1200 s
- Mission duration: 2700 s
- Minimum separation: 20 m
- Communication range: 100 m
- PoIs: 10
- Dynamic PoI spawning: enabled
- Reporting deadline: 10 s
- Initial simulator: deterministic/stochastic custom 2D/2.5D simulation
- Initial communication model: binary-range baseline

This baseline is for reproducible development and is not a claim about undisclosed organizer scoring formulas.

---

## 11. Audit Status

Phase 0 requirements audit: INITIALIZED

Next phase:

ARCHITECTURE + MATHEMATICAL INTERFACES

No large algorithm implementation should begin until the interfaces are defined.

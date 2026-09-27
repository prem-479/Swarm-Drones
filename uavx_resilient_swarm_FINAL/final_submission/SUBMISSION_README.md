# UAV-X Resilient BVLOS Swarm

## Submission Package

This package contains the UAV-X resilient BVLOS swarm simulation project.

### System

The simulator is a custom Python-based 2D/2.5D swarm simulation with:

- dynamic UAV state propagation
- communication graph generation
- autonomous task allocation
- relay selection and reconfiguration
- role management
- fault detection and recovery
- battery / return-to-home handling
- predictive collision avoidance
- acceleration-limited UAV motion
- geofencing
- mission and network metrics
- deterministic seeded scenarios
- real-time professional visualization

### Official Stage-1 Parameters

Operational area: 1000 × 1000 m

Maximum UAV speed: 5 m/s

UAV endurance: 1200 s

Maximum communication range: 100 m

Minimum UAV separation: 20 m

Maximum altitude: 100 m

Mission duration: 2700 s

PoI count: 10

Reporting deadline: 10 s

### Main Demo

`results/uavx_stage1_final_demo.mp4`

### Main Simulator

`scripts/professional_sim.py`

### Configuration

`configs/official_stage1.yaml`

### Source Architecture

Scenario
→ Simulation Engine
→ State Interface
→ Swarm Decision Engine
→ Motion Planner
→ Safety Layer
→ UAV Commands

### Validation

The final build should only be submitted after:

1. Python compilation succeeds.
2. Full pytest suite passes.
3. Physics sanity check passes.
4. Final demonstration video exists.

The supplied metrics and results are experimental outputs from the included simulator and should be interpreted as such.

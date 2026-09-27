# UAV-X Resilient BVLOS Swarm

Implementation of the UAV-X: Resilient BVLOS Swarm Challenge
for PUSHPAK Grand Challenge 2026.

## Initial Architecture

Scenario
  -> Simulation Engine
  -> State Interface
  -> Swarm Decision Engine
  -> Mission / Allocation / Relay / Fault / Energy
  -> Motion Planning
  -> Safety
  -> UAV Commands

## Initial Target

A fast custom 2D/2.5D simulation core suitable for
deterministic and Monte Carlo experiments.

## Official Baseline

- Operational area: 1000 m x 1000 m
- Maximum altitude: 100 m
- Maximum UAV speed: 5 m/s
- UAV endurance: 20 min
- Minimum separation: 20 m
- Communication range: 100 m
- Mission duration: 45 min
- PoI reporting deadline: 10 s

## Development Sequence

1. Requirements audit
2. Architecture
3. Core data model
4. Simulation engine
5. Communication graph
6. Relay management
7. Dynamic task allocation
8. Failure recovery
9. Energy management
10. Collision avoidance
11. Metrics
12. Monte Carlo validation
13. Gazebo / ROS 2 / PX4 integration

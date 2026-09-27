
UAV-X Resilient BVLOS Swarm — Technical Submission Report
System

The project implements a custom event-driven UAV swarm simulation with dynamic
communication, autonomous task allocation, relay management, role transitions,
failure handling, recovery, energy policy, and safety layers.

Stage-1 configuration
Area: 1000 m × 1000 m
UAV count: 10
Communication range: 100 m
Endurance: 1200 s
Mission duration: 2700 s
Minimum separation: 20 m
Maximum speed: 5 m/s
Maximum altitude: 100 m
Dynamic PoIs: 10
Reporting deadline: 10 s
Implemented autonomy
Communication

Dynamic distance-dependent communication graph with link quality,
packet-delivery probability, latency, availability, and topology analysis.

Relay management

Reactive relay selection is supplemented by predictive relay-backbone planning
for long-distance GCS-to-mission corridors.

Task allocation

Communication-aware task allocation considers mission priority,
travel feasibility, and communication constraints.

Fault handling

Failure evidence distinguishes packet loss from confirmed UAV failure.
Recovery provides replacement selection and reconfiguration primitives.

Energy

The stack tracks battery state and provides return-to-home decision logic.

Handover

Relay handover models:

PRE_HANDOVER →
RESERVE_DISPATCH →
RESERVE_APPROACH →
COMMUNICATION_OVERLAP →
ROLE_TRANSFER →
OLD_RELAY_RTH

Safety

Motion processing includes communication-preserving constraints,
collision-separation handling, speed limits, and geofence support.

Validation

The final software test suite passes all automated tests included in the
repository.

The simulation and evaluation artifacts are included under results/.

Evaluation integrity

The final evaluation JSON records the simulator's actual measured result.
No artificial mission-completion value or hidden scoring formula is inserted.

The current Stage-1 automated run should therefore be interpreted as a
development/evaluation result rather than a claim of perfect mission success.

Known engineering limitations

The current simulator is a custom 2D/2.5D model rather than a full PX4/Gazebo
flight-stack simulation.

Communication modeling is simulation-level rather than a physical RF/network
implementation.

The predictive relay planner is heuristic and should not be described as an
exact Steiner-tree optimizer.

The supplied Stage-1 evaluation currently demonstrates the implemented
decision/recovery stack but does not establish full PoI mission completion.

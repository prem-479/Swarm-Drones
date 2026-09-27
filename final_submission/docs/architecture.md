# UAV-X Architecture

```text
Scenario Engine
       |
       v
Simulation State Interface
       |
       v
Swarm Decision Engine
   +---+---+---+---+---+
   |   |   |   |   |   |
Mission Allocation Relay Fault Energy
   |   |   |   |   |
   +---+---+---+---+---+
           |
           v
Communication-Preserving Motion
           |
           v
Asymmetric Collision Avoidance
           |
           v
Geofence / Safety Guard
           |
           v
UAV Command Interface
      +----+----+
      |         |
 Custom Sim   ROS2/Gazebo/PX4

The controller is separated from the simulator adapter.

The custom simulator is the primary validation environment.

ROS 2, Gazebo, and PX4 remain optional integration targets.

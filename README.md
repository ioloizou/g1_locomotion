# g1_locomotion

Humanoid locomotion stack for a Unitree G1, built around a convex
single-rigid-body model predictive controller and a whole-body inverse-dynamics
QP, running against MuJoCo.

## How it fits together

```
                 /cmd_vel  (body-frame vx, vy, yaw rate)
                     |
                     v
  +--------------------------------------+        /mpc_solution
  |  g1_mpc  (C++, 25 Hz)                | ---------------------+
  |    gait scheduler  -> contact table  |                      |
  |    footstep planner -> landing spots |   states_horizon      |
  |    convex MPC       -> contact forces|   reference_horizon   |
  +--------------------------------------+   contacts, foot_plan|
                     ^                                          |
      /srbd_current  |                                          v
                     |                    +-----------------------------------+
                     +--------------------|  g1_mujoco_sim  (Python, 1 kHz)   |
                                          |    whole-body inverse dynamics    |
                                          |    swing trajectories             |
                                          |    MuJoCo plant                   |
                                          +-----------------------------------+
```

The MPC decides *what the contact forces should be and where the feet go*. The
whole-body controller decides *what joint torques produce them*, subject to the
robot's full dynamics, its friction cones and its joint and torque limits.

Both nodes read the same configuration file, so the numbers they have to agree
on cannot drift apart.

## Packages

| Package | Contents |
|---|---|
| `g1_description` | URDF, MuJoCo XML and meshes |
| `g1_msgs` | `SRBD_state`, `State`, `ContactPoint`, `FootPlan`, `Feet_reference` |
| `g1_mpc` | Gait scheduler, footstep planner, convex MPC, ROS node, self-test |
| `g1_mujoco_sim` | MuJoCo simulation, whole-body inverse dynamics, RViz output, config |

## Running it

```bash
make                                        # build every package
source /home/forest_ws/install/setup.bash

roslaunch g1_mujoco_sim g1_locomotion.launch          # stand
roslaunch g1_mujoco_sim walk.launch vx:=0.2           # walk forward
roslaunch g1_mujoco_sim walk.launch vx:=0.15 wz:=0.3  # walk and turn
```

Drive it live from another terminal:

```bash
rostopic pub -1 /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.2, y: 0.0}, angular: {z: 0.0}}'
```

Useful arguments: `headless:=true` (no MuJoCo window), `duration:=10` (stop
after ten seconds of simulated time), `rviz:=false`.

## Configuration

Everything tunable lives in
[`g1_mujoco_sim/config/g1_locomotion.yaml`](g1_mujoco_sim/config/g1_locomotion.yaml),
loaded into the parameter server by the launch file before either node starts.
Both nodes read it, so friction, gravity, gait timing and foot geometry are
stated once.

The `robot:` section is derived from the URDF rather than typed in:

```bash
rosrun g1_mujoco_sim extract_robot_params.py          # regenerate
rosrun g1_mujoco_sim extract_robot_params.py --check  # verify it is current
```

Run it after changing the model or the nominal configuration. It computes the
mass, the **centroidal** composite inertia, the CoM height above the soles, the
foot geometry and the spawn height, all with xbot2_interface.

## Checking things without the whole stack

```bash
rosrun g1_mujoco_sim pd_stand_demo.py --headless --duration 3   # model + contacts
/home/forest_ws/build/g1_mpc/devel/lib/g1_mpc/mpc_selftest      # MPC, gait, footsteps
```

`mpc_selftest` prints the contact forces the MPC produces for a handful of
states, the gait schedule tick by tick, and the footstep targets at several
speeds. It needs no ROS master, so it separates "the controller is wrong" from
"the simulation is wrong".

## Model and conventions

State (13): `[roll pitch yaw | x y z | wx wy wz | vx vy vz | g]`, orientation as
ZYX Euler angles of the base, position and linear velocity of the CoM, all in
the world frame. The last entry is a constant gravity augmentation that keeps
the affine gravity term inside the linear `A` matrix; its weight is zero.

Input (12): `[f_Lheel f_Ltoe f_Rheel f_Rtoe]`, each xyz, world frame.

Contact points are **always** ordered L-heel, L-toe, R-heel, R-toe. Frames are
x forward, y left, z up; the left foot is +y. Velocity commands are body-frame
and are rotated by the current heading before use.

Yaw is unwrapped everywhere it crosses a topic. The raw value from
`euler_from_matrix` folds at ±π while the reference keeps growing through a
turn, and the difference shows up as a spurious 2π error mid-turn.

## What it does today

Measured headless, 15 s of simulated time per command, from a standing start.
"holds" means it was still walking when the run ended.

| Command | Result |
|---|---|
| stand | holds |
| forward 0.1 m/s | holds |
| forward 0.2 m/s | walks ~7-12 s, then falls |
| forward 0.3 m/s | walks ~3 s, then falls |
| backward 0.1 m/s | holds |
| sidestep +/-0.1 m/s | holds |
| turn in place +/-0.3 rad/s | holds |
| forward 0.1 m/s + turn 0.2 rad/s | holds |
| forward 0.1 m/s + sidestep 0.05 m/s | holds |

Before this work the robot took a few forward steps and stopped: the gait table
disabled itself permanently at t = 1.5 s, and turning and sidestepping were not
implemented at all.

Runs vary a little because the two nodes are separate processes and their
timing is not locked together.

## Known limitations

- **Line feet.** Each foot is modelled as two contact points on its centreline,
  so the model has no roll moment at a single foot. Lateral balance comes almost
  entirely from where the next foot is placed.
- **Feed-forward footsteps.** The MPC assumes the scheduled footholds; the
  Raibert/capture-point feedback sits outside the QP.
- **Flat ground.** The friction pyramid and the swing trajectories assume a
  level floor at a known height.
- **Forward speed above ~0.2 m/s is not reliable.** The failure is a growing
  lateral oscillation combined with torque saturation at the hips. The next
  things to try are footstep placement from the divergent component of motion
  rather than a Raibert heuristic, and co-optimising the footholds inside the
  QP instead of feeding them forward.
- **No hardware bridge.** To run on a G1, replace the simulation: feed the state
  estimate into the same `/srbd_current` message, map each contact force to
  stance-leg torques with `tau_i = J_i^T (-f_i)`, and keep the swing-leg
  tracking.

## Reference

J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, S. Kim, "Dynamic Locomotion in
the MIT Cheetah 3 Through Convex Model-Predictive Control", IROS 2018.

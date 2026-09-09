# Changes

## Result

Headless, 15 s of simulated time per command, from a standing start:

| Command | Before | After |
|---|---|---|
| stand | falls | holds |
| forward 0.1 m/s | a few steps, then stops walking at t = 1.5 s | holds |
| forward 0.2 m/s | as above | walks 7-12 s |
| forward 0.3 m/s | as above | walks ~3 s |
| backward 0.1 m/s | not implemented | holds |
| sidestep +/-0.1 m/s | not implemented | holds |
| turn in place +/-0.3 rad/s | not implemented | holds |
| forward 0.1 + turn 0.2 rad/s | not implemented | holds |
| forward 0.1 + sidestep 0.05 m/s | not implemented | holds |

Everything below is a change from the previous state of this repository. The
headings follow the order the work was asked for.

## 1. Walking

### Bugs that made the robot fall

* **Whole-body base-orientation task had its frames the wrong way round.**
  `Cartesian("base", model, "world", "pelvis", ...)` asks for the pose of the
  *world in the pelvis*, which is the inverse rotation. Verified directly: with
  a +0.1 rad pitch and an upright reference, the old task commanded +5.0 rad/s²
  and the corrected one −5.0 rad/s². The error is invisible at zero yaw and
  destabilising as soon as the robot turns. Now `("base", model, "pelvis",
  "world", ...)`.

* **Single-rigid-body inertia was ~40x too small.** The MPC used
  `diag(0.082, 0.081, 0.0032)`, which is a single link, not the robot. The
  correct centroidal composite inertia is `diag(3.21, 3.05, 0.55)` with
  `Ixz = 0.43`. It is now computed from the URDF instead of typed in.

* **Euler-rate map used `Rz(psi)` instead of `Rz(psi)^T`.** Silent at yaw 0,
  divergent in a turn.

* **World inertia was `Rz^T I Rz` instead of `Rz I Rz^T`.** A different matrix
  as soon as the robot is not facing along x.

* **Whole-body controller summed contact forces with the wrong reshape.**
  `np.reshape(u, (3, 4)).sum(axis=1)` mixes x, y and z of different feet.
  It is `(4, 3)` summed down the rows.

* **Angular-acceleration reference used `sum(r_i x omega)`.** That is not a
  torque. It is now `Iw^-1 sum(r_i x f_i)` with `Iw = Rz I Rz^T`.

* **Stance-foot task reference was latched from the nominal configuration**
  rather than the state the robot was actually in. A few millimetres of
  disagreement asked a stiff task to move a foot already on the ground, and the
  QP answered by folding the knees: contact forces came out at 296 N against a
  334 N body weight, with 91 rad/s² of knee acceleration. The simulation now
  places the robot first and builds the stack from that state.

* **The robot spawned 5 mm inside the floor.** `robot/base_height` puts the
  contact *frames* on the ground, but the collision spheres extend below them.
  MuJoCo answered the penetration with roughly twice body weight on the first
  tick. The simulation now settles the model onto the ground from the geometry.

* **Raibert heuristic was wrong in three ways.** It multiplied the stance term
  by the MPC timestep (a 20x error), read `x_opt.row(10)` out of bounds on a
  10-row matrix whenever the sub-phase was zero, and shadowed its own output
  variables so the second footstep computation was discarded.

* **The MPC's assumed foot position was 3.5 cm too far forward.** Heel and toe
  are not symmetric about the foot frame (−5 cm and +12 cm), so their midpoint
  is not the frame. Every force distribution therefore put the real centre of
  pressure behind the intended one and drove the robot forwards while standing.

* **The velocity command was integrated into the reference while the feet were
  planted.** During the standing phase the CoM reference marched 24 cm away
  from the feet, so the robot was already toppling when the first foot lifted.
  Commanded motion is now accumulated only while the gait is actually stepping.

* **The whole-body controller tracked the MPC's rollout, which carries no
  feedback.** The rollout starts at the measured state, so one step ahead it is
  still essentially the measurement, and the CoM and trunk tasks had nothing to
  correct: the robot drifted away at close to the open-loop inverted-pendulum
  rate (measured 3.2 s^-1 against a natural 4.0 s^-1) with the controller doing
  nothing. The MPC now publishes its *reference* trajectory as well, and the
  whole-body controller tracks that.

* **Task weights were scale-mismatched.** Contact forces are order 100 N and
  accelerations order 1 m/s², so a force-tracking weight of 0.06 against a CoM
  weight of 8 silently outvoted balance: the QP produced 0.16 m/s² of CoM
  acceleration where 1.3 m/s² was available. Weights are now set so each term
  contributes comparably after residual scaling.

* **CoM gains demanded more than the support polygon can deliver.** With
  `kp = 300` any error over 4.5 mm saturates. Retuned, with the reasoning
  written down next to the numbers.

* **Stance-foot gains asked for more torque than the actuators have.** Hip roll
  (88 N·m), hip pitch and the shoulders were saturating during walking, which
  means the QP had stopped tracking anything. Softening them from `kp = 2000`
  to `800` took a 0.2 m/s run from 9.2 s to the full 12 s.

* **MPC state weights were badly proportioned.** Roll and pitch at 3000 against
  velocity at 100 meant a 0.05 rad tilt bought 1.0 m/s² of unwanted forward CoM
  acceleration, while a 0.3 m/s velocity error bought only 0.42 m/s² of braking.
  Attitude is now weighted an order of magnitude lower and velocity five times
  higher.

### New behaviour needed to walk at all

* **Lateral sway in the reference.** A straight-line CoM reference is only
  correct in double support; during single support it fights the very weight
  shift the step depends on. The reference now leans towards the support
  centroid, crossing over during the double-support interval.

* **Startup delay.** A walk command must persist for 1 s before the first step,
  so the robot is not asked to lift a foot while still centred between them.

* **Gait phase restarts when walking begins,** so the first liftoff is a whole
  double-support interval away instead of possibly one tick away.

* **Early touchdown detection.** Feet routinely reach the ground before the
  schedule says so. Continuing to drive one into ground it has already reached
  launched the robot upwards. Contact is now accepted from 60 % of the swing
  onwards, latched for the rest of that swing.

* **Yaw is unwrapped** in the state the MPC receives.

* **Swing foot lands level and yawed with the body** instead of world-aligned.

* **Raibert split into feed-forward and feedback.** The half-stance travel term
  now uses the commanded velocity, not the measured one. A biped's lateral
  velocity oscillates by design, and pushing that oscillation through a term as
  large as half a stance duration widened the step every cycle until the robot
  stepped itself over. This single change took a 0.05 m/s walk from 3.2 s to
  11.4 s.

## 2. Gait

Replaced a hardcoded 20-row contact table, indexed modulo the horizon length
into a schedule that was not periodic over that length, with a
`GaitScheduler`:

* Integer-tick based, aligned to MPC solve ticks, so contact transitions cannot
  land between two solves.
* Period, stance duration and leg offset are configuration, not code.
* Standing and walking are proper modes, entered only when both feet are down.
* Exposes contact flags anywhere in the horizon, swing phase, liftoff ticks and
  time to touchdown.

The old table stopped walking permanently at t = 1.5 s (`gait_phase >= 30` set
`is_standing`), and the two tables it switched between were not phase
consistent, so the first one never produced a left-support phase at all. That
is why the robot took a few steps and stopped.

The simulation no longer runs its own swing clock. It follows the plan
published by the MPC, so the two cannot disagree about which foot is carrying
load.

## 3. Configuration

* New `g1_mujoco_sim/config/g1_locomotion.yaml`: one file, loaded into the
  parameter server by the launch file before either node starts. Both the C++
  MPC and the Python simulation read it, so friction, gravity, gait timing and
  foot geometry are stated exactly once. Previously friction appeared as `0.8`
  in two places and `1.0` in the MuJoCo model, and gravity as `-9.80665` in the
  controllers against MuJoCo's default `-9.81`.
* New `scripts/extract_robot_params.py`: derives mass, centroidal inertia, CoM
  height, foot geometry and spawn height from the URDF with xbot2_interface and
  writes them into the config. `--check` fails if the config has gone stale;
  wired up as `make check-config`.
* The nominal configuration moved from a bare list to
  `nominal_configuration.py`, keyed by joint name, with the joint order
  asserted against MuJoCo at start-up.
* The MPC's parameter loading is written once and read through a small source
  interface, so the ROS parameter server and the YAML file cannot describe
  different parameter sets.
* Every parameter is validated on load; a missing key names itself and stops
  the node rather than silently defaulting to something the other node
  disagrees with.

## 4. Footstep planning

`FootstepPlanner` replaces the inline heuristic:

* Nominal placement in the **heading frame**, so the stance offset follows the
  robot round a turn. The old code added a fixed ±0.118 m in world y, which
  made turning impossible.
* Capture-point gain `sqrt(h/g)` rather than an arbitrary 0.02.
* Centrifugal term for turning while translating.
* Three clamps: on the feedback correction alone, on the step relative to the
  predicted hip, and a minimum lateral offset so the swing leg can never cross
  the stance leg.
* Future footholds are planned against the reference trajectory and fed into
  the MPC's horizon, rather than assuming the feet stay where they are.

## 5. Repository structure

```
g1_mpc/                      g1_mujoco_sim/
  include/g1_mpc/*.hpp         config/g1_locomotion.yaml
  src/mpc.cpp                  launch/{g1_locomotion,walk}.launch
     gait_scheduler.cpp        scripts/{g1_simulation_node,
     footstep_planner.cpp               extract_robot_params,
     parameters.cpp                     pd_stand_demo}.py
     mpc_node.cpp              src/g1_mujoco_sim/*.py
  test/mpc_selftest.cpp
```

* `mpc_to_wbid_node.cpp` did `#include "mpc.cpp"`, compiling the MPC twice and
  linking only by accident. It now links the library properly.
* `add_library(g1_mpc src/mpc.cpp src/)` listed a directory as a source file.
* Hardcoded `/home/forest_ws/install` paths removed from CMake.
* `g1_msgs` was missing from `find_package` despite being included.
* Headers moved under `include/g1_mpc/`.
* Python sources became a real package under `src/g1_mujoco_sim/`; nodes are in
  `scripts/`. (Installed directly rather than through `catkin_python_setup()`,
  which passes a `--install-layout` flag setuptools has removed.)
* The launch file pointed at `$(find g1_mujoco_sim)/g1_srbd_full.rviz`; the
  file is in `config/`. Two launch files now, with `headless`, `duration` and
  `rviz` arguments.
* RViz config updated to the topics that are actually published.
* Dead code removed: `run_simulation.py` (imported a module that is not in this
  workspace), `PD_controller.py`, `config.py`. The useful part of the PD demo
  survives as `pd_stand_demo.py`, now with gravity compensation.
* The standalone Python SRBD prototype in `g1_mpc/scripts/` is untouched and
  documented as not part of the running stack.
* `Makefile` builds in dependency order and gained `test` and `check-config`.
* `cmake_minimum_required` bumped to 3.10 to silence deprecation warnings.

## 6. Other fixes between the pieces

* **The MuJoCo model had no `<option>` block**, so it ran at the default
  gravity and integrator regardless of what the controllers assumed. Gravity,
  timestep, integrator and cone are now explicit and match the config.
* **Foot friction is set from the config at load time**, so the friction cones
  the controllers respect are the ones the ground enforces.
* **A new OSQP solver was constructed on every MPC solve.** `isInitialized()`
  was therefore always false, the update branch was dead code, and warm
  starting never happened. The solver is now built once, with a fixed Hessian
  sparsity pattern so it can be updated in place.
* **The QP result was used without checking the solver status.** A failed solve
  published whatever was in the buffer. There is now a status check and a
  static weight-split fallback, reported on the `solver_ok` field.
* **The MPC ran in a free-running 400 Hz loop** re-solving a 120-variable QP
  regardless of whether anything had changed. It now solves exactly once per
  gait tick, at 25 Hz.
* **`pal_statistics` registrations were created and destroyed inside the
  control loop.** Solve time is published on a plain topic instead.
* **The visualiser constructed a fresh `rospy.Publisher` inside every publish
  call**, at 1 kHz, across eight topics. Each one registers with the master.
  Publishers are created once and markers are throttled to 50 Hz.
* **Discretisation is now exact.** `A` is nilpotent for this state layout, so
  the exponential series terminates; the previous forward-Euler `A*dt` and
  `B*dt` are replaced by the exact zero-order hold at the same cost.
* **`State.gravity` was `float32`** in a message whose other fields are
  `float64`.
* **The simulation crashed the process instead of stopping.** qpOASES
  segfaults when handed the QP of an already-fallen robot; there is now a fall
  and divergence check before the solve.
* **Torque saturation is reported.** It means the QP asked for something the
  robot cannot do, and it was happening silently.
* **`mpc_selftest`** exercises the MPC, gait scheduler and footstep planner
  without a ROS master, reading the same config file the stack runs on.

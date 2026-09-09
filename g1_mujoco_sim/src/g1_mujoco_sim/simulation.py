"""MuJoCo simulation driving the whole-body controller from the MPC plan.

Loop structure, one physics tick:
  1. read MuJoCo state, convert to the xbot convention, update the model;
  2. apply the gait state from the latest MPC plan (stance/swing per leg, swing
     trajectory by phase);
  3. solve the whole-body QP against the MPC's state and force references;
  4. integrate MuJoCo with the resulting torques;
  5. publish the measured single-rigid-body state at the MPC rate.

The gait comes entirely from the plan. Running an independent swing timer here,
as an earlier version did, meant the planner and the simulation disagreed about
which foot was carrying load, which is a reliable way to fall over.
"""

import mujoco
import numpy as np
import rospy
import tf.transformations as tf_trans

from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist

from g1_msgs.msg import ContactPoint, Feet_reference, SRBD_state, State

from g1_mujoco_sim.parameters import Parameters
from g1_mujoco_sim.swing_trajectory import SwingTrajectory
from g1_mujoco_sim.visualization import Visualizer
from g1_mujoco_sim.whole_body_id import LEFT, RIGHT, WholeBodyID

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class G1Simulation(object):
    def __init__(self, urdf_path, model_path, q_nominal, joint_names,
                 headless=False, duration=None, params=None):
        self.params = params or Parameters()
        self.contact_frames = self.params.contact_frames
        self.foot_frames = self.params.foot_frames

        with open(urdf_path, "r") as handle:
            self.urdf = handle.read()

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.timestep = float(self.params.simulation.timestep)
        self.model.opt.timestep = self.timestep
        # Keep the plant's gravity and friction identical to what the planner
        # assumes; a mismatch here shows up as a controller that is subtly
        # wrong in a way no amount of gain tuning fixes.
        self.model.opt.gravity[:] = self.params.gravity_vector
        self._apply_friction(float(self.params.physics.friction_coefficient))

        self.mpc_dt = float(self.params.mpc.dt)
        self.publish_every = max(1, int(round(self.mpc_dt / self.timestep)))

        self.q_nominal = np.asarray(q_nominal, dtype=float).copy()
        self.joint_names = list(joint_names)
        self._check_joint_order()
        self._foot_geoms = self._collect_foot_geoms()

        self.wbid = WholeBodyID(self.urdf, self.timestep, self.q_nominal, self.params)

        # Place the robot before building the stack. The stance-foot tasks latch
        # their reference from the model state at setup, so that state has to be
        # the one the robot is actually in: a few millimetres of disagreement
        # asks a 3000-gain task to move a foot that is already on the ground,
        # and the QP answers by folding the knees.
        self._reset_state()
        self.wbid.q, self.wbid.dq = self._xbot_state()
        self.wbid.setup()

        swing_height = float(self.params.gait.swing_height)
        self.swing = [SwingTrajectory(swing_height), SwingTrajectory(swing_height)]
        self.swing_duration = self._swing_duration()
        self._swing_active = [False, False]
        self._early_touchdown = [False, False]

        # Latest plan from the MPC.
        self.plan_state = np.zeros(13)
        self.plan_forces = np.zeros((4, 3))
        self.plan_feet = None
        self.plan_message = SRBD_state()
        self.have_plan = False

        self.sim_time = 0.0
        self.step_index = 0
        self.duration = duration
        self.fallen = False
        self.wbid_solve_time = 0.0

        self.viewer = None
        if not headless:
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.visualizer = Visualizer(self.urdf, self.contact_frames)

        self.pub_state = rospy.Publisher("/srbd_current", SRBD_state, queue_size=1)
        self.pub_clock = rospy.Publisher("/simulation_time", Clock, queue_size=1)
        self.pub_feet_reference = rospy.Publisher("/feet_ref_pos", Feet_reference,
                                                  queue_size=1)
        self.sub_plan = rospy.Subscriber("/mpc_solution", SRBD_state,
                                         self._on_plan, queue_size=1)
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=True)

    # ------------------------------------------------------------------ setup

    def _apply_friction(self, mu):
        """Set the sliding friction of every geom to the planned value."""
        self.model.geom_friction[:, 0] = mu

    def _check_joint_order(self):
        """MuJoCo and xbot must agree on the actuated joint ordering.

        Nothing in either format guarantees it; the state and torque vectors are
        passed between them by index, so a mismatch would silently drive the
        wrong joints.
        """
        mujoco_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        if mujoco_names != self.joint_names:
            raise RuntimeError(
                "MuJoCo actuator order does not match the model joint order.\n"
                "  mujoco: %s\n  model:  %s" % (mujoco_names, self.joint_names))

    def _collect_foot_geoms(self):
        """Collision spheres belonging to each foot, ordered left then right."""
        geoms = [[], []]
        for leg, frame in enumerate(("left_ankle_roll_link", "right_ankle_roll_link")):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame)
            if body < 0:
                raise RuntimeError("no body named %s in the MuJoCo model" % frame)
            for geom in range(self.model.ngeom):
                if self.model.geom_bodyid[geom] != body:
                    continue
                if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_SPHERE:
                    continue
                geoms[leg].append(geom)
        if not all(geoms):
            raise RuntimeError("no foot collision spheres found in the MuJoCo model")
        return geoms

    def _measured_contact(self):
        """Which feet are actually touching the ground right now.

        A real robot would read this from force sensors. It matters because the
        swing foot regularly reaches the ground before the schedule says it
        should: if the controller still believes that foot is swinging, it keeps
        driving it downwards into a surface that is already pushing back, and
        the robot gets launched off it.
        """
        touching = [False, False]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for leg in (LEFT, RIGHT):
                if contact.geom1 in self._foot_geoms[leg] or \
                        contact.geom2 in self._foot_geoms[leg]:
                    touching[leg] = True
        return touching

    def _reset_state(self):
        """Place the robot at the nominal configuration with its soles on the ground."""
        q = self.q_nominal.copy()
        q[2] = float(self.params.robot.base_height)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = q[0:3]
        # xbot stores the quaternion as (x, y, z, w), MuJoCo as (w, x, y, z).
        self.data.qpos[3] = q[6]
        self.data.qpos[4:7] = q[3:6]
        self.data.qpos[7:] = q[7:]
        mujoco.mj_forward(self.model, self.data)

        # robot/base_height comes from forward kinematics and puts the contact
        # *frames* on the ground, but the collision spheres extend below them.
        # Spawning with the feet a few millimetres inside the floor makes the
        # contact solver answer with roughly twice the robot's weight on the
        # first tick, which throws the base before the controller has run once.
        self.data.qpos[2] -= self._lowest_contact_height()
        mujoco.mj_forward(self.model, self.data)

        self.yaw_unwrapped = 0.0
        self._last_yaw = 0.0

    def _lowest_contact_height(self):
        """Height of the lowest point of any collidable sphere, world frame."""
        lowest = None
        for geom in range(self.model.ngeom):
            if self.model.geom_contype[geom] == 0 and \
                    self.model.geom_conaffinity[geom] == 0:
                continue
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_SPHERE:
                continue
            bottom = self.data.geom_xpos[geom][2] - self.model.geom_size[geom][0]
            lowest = bottom if lowest is None else min(lowest, bottom)
        if lowest is None:
            raise RuntimeError("no collidable foot spheres found in the MuJoCo model")
        return lowest

    def _swing_duration(self):
        gait = self.params.gait
        return (int(gait.period_ticks) - int(gait.stance_ticks)) * self.mpc_dt

    # -------------------------------------------------------------- callbacks

    @staticmethod
    def _state_vector(state):
        return np.array([
            state.orientation.x, state.orientation.y, state.orientation.z,
            state.position.x, state.position.y, state.position.z,
            state.angular_velocity.x, state.angular_velocity.y,
            state.angular_velocity.z,
            state.linear_velocity.x, state.linear_velocity.y,
            state.linear_velocity.z,
            state.gravity,
        ])

    def _on_plan(self, msg):
        if len(msg.reference_horizon) < 2 or len(msg.contacts) != 4:
            return
        # Track the *reference* one MPC step ahead, not the MPC's rollout.
        #
        # The rollout starts at the measured state, so one step ahead it is
        # still essentially the measurement: tracking it leaves the CoM and
        # trunk tasks with no error, and the robot drifts away at close to the
        # open-loop inverted-pendulum rate with the controller doing nothing.
        # The reference is where the robot is supposed to be, so tracking it
        # gives those tasks something to correct, while the MPC's forces below
        # still carry the feed-forward and the contact schedule.
        self.plan_state = self._state_vector(msg.reference_horizon[1])
        for i, contact in enumerate(msg.contacts):
            self.plan_forces[i] = (contact.force.x, contact.force.y, contact.force.z)
        self.plan_feet = list(msg.foot_plan) if msg.foot_plan else None
        self.plan_message = msg
        self.have_plan = True

    def send_command(self, linear_x, linear_y, angular_z):
        command = Twist()
        command.linear.x = linear_x
        command.linear.y = linear_y
        command.angular.z = angular_z
        self.pub_cmd.publish(command)

    # ------------------------------------------------------------------ state

    def _xbot_state(self):
        """MuJoCo state in the xbot convention.

        Base linear velocity is world frame in MuJoCo but local frame in
        xbot/pinocchio; the angular velocity is already local in both.
        """
        q = self.data.qpos.copy()
        q_xbot = q.copy()
        q_xbot[3:6] = q[4:7]
        q_xbot[6] = q[3]

        rotation = tf_trans.quaternion_matrix(q_xbot[3:7])[0:3, 0:3]
        v = self.data.qvel.copy()
        v_xbot = v.copy()
        v_xbot[0:3] = rotation.T @ v[0:3]
        return q_xbot, v_xbot

    def _contact_positions(self):
        return np.array([self.wbid.model.getPose(f).translation
                         for f in self.contact_frames])

    def _update_yaw(self, rotation):
        """Accumulate yaw so it does not wrap at +-pi.

        The reference yaw grows without bound during a turn; feeding a wrapped
        measurement against it creates a spurious 2*pi error the moment the
        robot crosses the branch cut.
        """
        yaw = tf_trans.euler_from_matrix(rotation)[2]
        self.yaw_unwrapped += wrap_to_pi(yaw - self._last_yaw)
        self._last_yaw = yaw
        return self.yaw_unwrapped

    # -------------------------------------------------------------------- gait

    def _apply_gait(self):
        """Follow the plan: set stance/swing per leg and drive the swing feet."""
        if self.plan_feet is None:
            return

        touching = self._measured_contact()
        early_phase = float(self.params.gait.early_touchdown_phase)

        for leg in (LEFT, RIGHT):
            foot = self.plan_feet[leg]
            planned_contact = bool(foot.in_contact)

            # Accept the ground when it arrives ahead of schedule, but only
            # once the foot is far enough through its swing that this is a
            # landing rather than a scuff on lift-off. The decision is latched
            # for the rest of the swing so a bouncing contact cannot make the
            # leg flicker between stance and swing every millisecond.
            if planned_contact:
                self._early_touchdown[leg] = False
            elif touching[leg] and float(foot.swing_phase) >= early_phase:
                self._early_touchdown[leg] = True
            in_contact = planned_contact or self._early_touchdown[leg]

            if not in_contact and not self._swing_active[leg]:
                # Liftoff: anchor the trajectory at the foot's actual position
                # so there is no jump at the start of the swing.
                start = self.wbid.model.getPose(self.foot_frames[leg]).translation
                self.swing[leg].set_endpoints(start, self._landing(foot))
                self._swing_active[leg] = True
            elif in_contact:
                self._swing_active[leg] = False

            self.wbid.set_contact_state(leg, in_contact)

            if not in_contact:
                # Re-aim at the latest target. The trajectory is a function of
                # phase, so a small change in target moves the foot smoothly
                # instead of restarting the swing.
                self.swing[leg].set_endpoints(self.swing[leg].start,
                                              self._landing(foot))
                phase = float(foot.swing_phase)
                self.wbid.set_swing_reference(
                    leg,
                    self.swing[leg].position(phase),
                    self.swing[leg].velocity(phase, self.swing_duration),
                    self.swing[leg].acceleration(phase, self.swing_duration),
                    yaw=self.plan_state[2])

    @staticmethod
    def _landing(foot):
        return np.array([foot.landing_position.x, foot.landing_position.y,
                         foot.landing_position.z])

    # -------------------------------------------------------------------- step

    def step(self):
        q, v = self._xbot_state()
        self.wbid.update_model(q, v)

        # Stop before solving, not after. Once the robot is on its side the
        # whole-body QP is being handed a hopeless problem, and the solver
        # crashes the process rather than returning a bad answer.
        if self._check_failed():
            return

        self._apply_gait()

        if self.have_plan:
            self.wbid.set_reference(self.plan_state, self.plan_forces)
        else:
            # Hold the nominal CoM until the first plan arrives.
            self.wbid.set_reference(self._standing_state(), self._standing_forces())

        start = rospy.Time.now()
        self.wbid.solve()
        tau = self.wbid.inverse_dynamics()
        self.wbid_solve_time = (rospy.Time.now() - start).to_sec()

        # The first six entries are the unactuated floating base.
        joint_torque = tau[6:]
        self._report_saturation(joint_torque)
        self.data.ctrl[:] = joint_torque
        mujoco.mj_step(self.model, self.data)

        self.sim_time += self.timestep
        self.step_index += 1

        q, v = self._xbot_state()
        self.wbid.update_model(q, v)

        self._publish_clock()
        if self.step_index % self.publish_every == 0:
            self._publish_state()
        if self.visualizer.due(self.sim_time):
            self._publish_visualization()

        self._check_failed()

    def _check_failed(self):
        """True once the run is over: the robot fell, or the state went bad."""
        if self.fallen:
            return True
        if not (np.all(np.isfinite(self.data.qpos)) and
                np.all(np.isfinite(self.data.qvel))):
            rospy.logerr("g1_mujoco_sim: simulation state diverged")
            self.fallen = True
            return True
        com = self.wbid.model.getCOM()
        if com[2] < float(self.params.simulation.fall_height):
            self.fallen = True
            return True
        # A trunk past this tilt is never recovered by this controller, and
        # letting it continue only risks the solver.
        upright = self.wbid.model.getPose("pelvis").linear[2, 2]
        if upright < 0.5:
            rospy.logerr("g1_mujoco_sim: trunk tipped past 60 degrees")
            self.fallen = True
            return True
        return False

    def _report_saturation(self, torque):
        """Warn when a joint is against its effort limit.

        Saturation is worth surfacing rather than silently clipping: it means
        the whole-body QP asked for something the robot cannot do, and whatever
        it was tracking is no longer being tracked.
        """
        limits = self.wbid.torque_limits[6:]
        hit = np.abs(torque) > 0.95 * limits
        if not hit.any():
            return
        names = [self.joint_names[i] for i in np.flatnonzero(hit)]
        rospy.logwarn_throttle(
            1.0, "g1_mujoco_sim: torque limit reached at %s" % ", ".join(names))

    def _standing_state(self):
        com = self.wbid.model.getCOM()
        state = np.zeros(13)
        state[3:6] = com
        state[5] = float(self.params.robot.com_height)
        state[12] = float(self.params.physics.gravity)
        return state

    def _standing_forces(self):
        weight = float(self.params.robot.mass) * abs(float(self.params.physics.gravity))
        forces = np.zeros((4, 3))
        forces[:, 2] = weight / 4.0
        return forces

    # --------------------------------------------------------------- publish

    def _publish_clock(self):
        message = Clock()
        message.clock = rospy.Time.from_sec(self.sim_time)
        self.pub_clock.publish(message)

    def _publish_state(self):
        model = self.wbid.model
        rotation = model.getPose("pelvis").linear
        roll, pitch, _ = tf_trans.euler_from_matrix(rotation)
        yaw = self._update_yaw(rotation)

        message = SRBD_state()
        message.header.stamp = rospy.Time.from_sec(self.sim_time)
        message.header.frame_id = "world"
        message.solver_ok = True

        state = State()
        state.trajectory_index = 0
        state.orientation.x = roll
        state.orientation.y = pitch
        state.orientation.z = yaw
        com = model.getCOM()
        state.position.x, state.position.y, state.position.z = com
        omega = model.getVelocityTwist("pelvis")[3:6]
        state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z = omega
        velocity = model.getCOMVelocity()
        state.linear_velocity.x, state.linear_velocity.y, state.linear_velocity.z = velocity
        state.gravity = float(self.params.physics.gravity)
        message.states_horizon.append(state)

        positions = self._contact_positions()
        forces = self.wbid.contact_forces
        for i, frame in enumerate(self.contact_frames):
            point = ContactPoint()
            point.name = frame
            point.position.x, point.position.y, point.position.z = positions[i]
            point.force.x, point.force.y, point.force.z = forces[i]
            point.active = self.wbid.in_contact(i // 2)
            message.contacts.append(point)

        self.pub_state.publish(message)

    def _publish_visualization(self):
        stamp = rospy.Time.from_sec(self.sim_time)
        self.visualizer.publish_joint_state(stamp, self.wbid.model.getJointPosition())
        self.visualizer.publish_contact_forces(stamp, self.wbid.contact_forces)
        if not self.have_plan:
            return
        self.visualizer.publish_srbd(stamp, self.wbid.inertia_body, self.plan_message)
        self.visualizer.publish_contacts(stamp, self.plan_message)
        self.visualizer.publish_footstep_targets(stamp, self.plan_message)
        self.visualizer.publish_com_trajectory(stamp,
                                               self.plan_message.states_horizon)
        for leg in (LEFT, RIGHT):
            if self._swing_active[leg]:
                self.visualizer.publish_swing_trajectory(stamp,
                                                         self.swing[leg].sample())
                break

        reference = Feet_reference()
        reference.header.stamp = stamp
        reference.header.frame_id = "world"
        for leg, frame in enumerate(self.foot_frames):
            point = ContactPoint()
            point.name = frame
            target = self.swing[leg].end if self._swing_active[leg] else \
                self.wbid.model.getPose(frame).translation
            point.position.x, point.position.y, point.position.z = target
            point.active = self.wbid.in_contact(leg)
            reference.feet_positions.append(point)
        self.pub_feet_reference.publish(reference)

    # ------------------------------------------------------------------- loop

    def run(self):
        rate = rospy.Rate(1.0 / self.timestep)
        realtime = bool(self.params.simulation.realtime)

        while not rospy.is_shutdown():
            if self.duration is not None and self.sim_time >= self.duration:
                rospy.loginfo("g1_mujoco_sim: reached %.2f s, stopping",
                              self.duration)
                break
            if self.fallen:
                rospy.logerr("g1_mujoco_sim: robot fell at t = %.2f s", self.sim_time)
                break
            if self.viewer is not None and not self.viewer.is_alive:
                break

            self.step()

            if self.viewer is not None and self.step_index % 20 == 0:
                self.viewer.render()
            if realtime:
                rate.sleep()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

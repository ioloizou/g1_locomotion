"""Whole-body inverse dynamics: turn the MPC plan into joint torques.

The stack tracks the CoM trajectory and base attitude the MPC asked for, holds
the stance feet still, drives the swing feet along their trajectory, and asks
for the ground reaction forces the MPC distributed, all subject to the robot's
dynamics, friction cones, and joint/torque limits.
"""

import numpy as np
import tf.transformations as tf_trans

from xbot2_interface import pyxbot2_interface as xbi

import pyopensot as pysot
from pyopensot.tasks.acceleration import (AngularMomentum, Cartesian, CoM,
                                          DynamicFeasibility, Postural)
from pyopensot.constraints.acceleration import (JointLimits, TorqueLimits,
                                                VelocityLimits)
from pyopensot.constraints.force import FrictionCone, WrenchLimits
from pyopensot.tasks import MinimizeVariable
from pyopensot.variables import Torque

LEFT = 0
RIGHT = 1


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class WholeBodyID(object):
    """Acceleration-and-force QP over qddot and the four contact forces."""

    def __init__(self, urdf, dt, q_init, params):
        self.dt = float(dt)
        self.params = params
        self.model = xbi.ModelInterface2(urdf)

        self.qmin, self.qmax = self.model.getJointLimits()
        self.dqmax = self.model.getVelocityLimits()
        self.torque_limits = self.model.getEffortLimits()

        self.q = np.asarray(q_init, dtype=float).copy()
        self.dq = np.zeros(self.model.nv)

        self.contact_frames = params.contact_frames
        self.foot_frames = params.foot_frames
        self.mass = float(params.robot.mass)
        self.inertia_body = params.inertia
        self.gravity = params.gravity_vector

        # Per-leg contact state, so tasks are only toggled on an actual
        # transition. Resetting a stance task every tick would re-anchor it to
        # wherever the foot has drifted to, which removes the very error the
        # task exists to correct.
        self._leg_in_contact = [True, True]
        self._initialised = False

        self.contact_forces = [np.zeros(3) for _ in self.contact_frames]
        self.ddq = np.zeros(self.model.nv)

    # ------------------------------------------------------------------ setup

    def setup(self):
        self.model.setJointPosition(self.q)
        self.model.setJointVelocity(self.dq)
        self.model.update()

        variables = {"qddot": self.model.nv}
        for frame in self.contact_frames:
            variables[frame] = 3
        self.variables = pysot.OptvarHelper(variables)
        qddot = self.variables.getVariable("qddot")

        cfg = self.params.wbid

        # --- CoM ------------------------------------------------------------
        com_cfg = cfg.com
        self.com = CoM(self.model, qddot)
        self.com.setKp(np.diag(com_cfg["kp"]))
        self.com.setKd(np.diag(com_cfg["kd"]))
        self.com.setLambda(1.0, 1.0)
        self.com_reference = self.com.getReference()[0].copy()

        # --- base attitude ---------------------------------------------------
        # distal link first, then base link: this is the pose of the pelvis in
        # the world, not the pose of the world in the pelvis. With the two
        # swapped the task tracks the inverse rotation, which is invisible at
        # zero yaw and actively destabilising as soon as the robot turns.
        base_cfg = cfg.base_orientation
        self.base = Cartesian("base", self.model, "pelvis", "world", qddot)
        self.base.setKp(np.diag(np.concatenate([np.zeros(3), base_cfg["kp"]])))
        self.base.setKd(np.diag(np.concatenate([np.zeros(3), base_cfg["kd"]])))
        self.base.setLambda(1.0, 1.0)

        # --- feet ------------------------------------------------------------
        stance_cfg = cfg.stance_foot
        swing_cfg = cfg.swing_foot
        self.contact_tasks = []
        self.swing_tasks = []
        for frame in self.foot_frames:
            contact = Cartesian(frame + "_stance", self.model, frame, "world", qddot)
            contact.setKp(np.diag(stance_cfg["kp"]))
            contact.setKd(np.diag(stance_cfg["kd"]))
            contact.setLambda(1.0, 1.0)
            self.contact_tasks.append(contact)

            swing = Cartesian(frame + "_swing", self.model, frame, "world", qddot)
            swing.setKp(np.diag(swing_cfg["kp"]))
            swing.setKd(np.diag(swing_cfg["kd"]))
            swing.setLambda(1.0, 1.0)
            self.swing_tasks.append(swing)

        # --- posture and angular momentum ------------------------------------
        posture_cfg = cfg.posture
        posture = Postural(self.model, qddot)
        posture.setKp(np.eye(self.model.nv) * posture_cfg["kp"])
        posture.setKd(np.eye(self.model.nv) * posture_cfg["kd"])
        posture.setLambda(1.0, 1.0)
        # Only regularise the upper body. Constraining the legs towards a fixed
        # posture fights the foot tasks that are doing the actual walking.
        upper_body = self._upper_body_indices()

        momentum_cfg = cfg.angular_momentum
        angular_momentum = AngularMomentum(self.model, qddot)
        angular_momentum.setMomentumGain(np.eye(3) * momentum_cfg["kd"])
        angular_momentum.setLambda(1.0)

        force_variables = [self.variables.getVariable(f) for f in self.contact_frames]
        torques = Torque(self.model, qddot, self.contact_frames, force_variables)

        reg_qddot = MinimizeVariable("acceleration", qddot)

        # --- wrench limits and force tracking --------------------------------
        fz_min = float(self.params.physics.fz_min)
        fz_max = float(self.params.physics.fz_max)
        self._stance_wrench_lower = np.array([-fz_max, -fz_max, fz_min])
        self._stance_wrench_upper = np.array([fz_max, fz_max, fz_max])
        self._swing_wrench = np.zeros(3)

        self.wrench_limits = []
        self.wrench_tasks = []
        for frame in self.contact_frames:
            var = self.variables.getVariable(frame)
            self.wrench_limits.append(
                WrenchLimits(frame, self._stance_wrench_lower.copy(),
                             self._stance_wrench_upper.copy(), var))
            self.wrench_tasks.append(MinimizeVariable(frame + "_force", var))

        # --- assemble --------------------------------------------------------
        stack = com_cfg["weight"] * self.com
        stack += base_cfg["weight"] * (self.base % [3, 4, 5])
        stack += posture_cfg["weight"] * (posture % upper_body)
        stack += momentum_cfg["weight"] * angular_momentum
        stack += float(cfg.acceleration_regularisation) * reg_qddot
        stack += float(cfg.torque_regularisation) * MinimizeVariable("torque", torques)

        for i in range(len(self.foot_frames)):
            stack += stance_cfg["weight"] * self.contact_tasks[i]
            stack += swing_cfg["weight"] * self.swing_tasks[i]

        force_weight = float(cfg.force_tracking_weight)
        for task in self.wrench_tasks:
            stack += force_weight * task

        stack = pysot.AutoStack(stack) << DynamicFeasibility(
            "floating_base_dynamics", self.model, qddot, force_variables,
            self.contact_frames)
        stack = stack << JointLimits(self.model, qddot, self.qmax, self.qmin,
                                     10.0 * self.dqmax, self.dt)
        stack = stack << VelocityLimits(self.model, qddot, self.dqmax, self.dt)
        stack = stack << TorqueLimits(self.model, qddot, force_variables,
                                      self.contact_frames, self.torque_limits)

        mu = float(self.params.physics.friction_coefficient)
        for i, frame in enumerate(self.contact_frames):
            # World-aligned contact normal: the ground is flat and level here.
            cone = FrictionCone(frame, self.variables.getVariable(frame),
                                self.model, (np.eye(3), mu))
            stack = stack << cone << self.wrench_limits[i]

        self.stack = stack
        self.solver = pysot.iHQP(self.stack)

        # Start in double support with both feet anchored where they are.
        for leg in (LEFT, RIGHT):
            self._apply_contact_state(leg, True)
        self._initialised = True

    def _upper_body_indices(self):
        """Velocity indices of the waist and arm joints."""
        indices = []
        for name in self.model.getJointNames():
            if any(tag in name for tag in ("waist", "shoulder", "elbow", "wrist")):
                indices.append(self.model.getVIndexFromVName(name))
        return sorted(indices)

    # ------------------------------------------------------------- gait state

    def set_contact_state(self, leg, in_contact):
        """Switch a leg between stance and swing, only on an actual change."""
        if self._initialised and self._leg_in_contact[leg] == in_contact:
            return
        self._apply_contact_state(leg, in_contact)

    def _apply_contact_state(self, leg, in_contact):
        self._leg_in_contact[leg] = in_contact
        self.contact_tasks[leg].setActive(in_contact)
        self.swing_tasks[leg].setActive(not in_contact)
        if in_contact:
            # Latch the stance reference to where the foot actually landed.
            self.contact_tasks[leg].reset()
        else:
            self.swing_tasks[leg].reset()

        lower = self._stance_wrench_lower if in_contact else self._swing_wrench
        upper = self._stance_wrench_upper if in_contact else self._swing_wrench
        for point in (2 * leg, 2 * leg + 1):
            self.wrench_limits[point].setWrenchLimits(lower.copy(), upper.copy())

    def in_contact(self, leg):
        return self._leg_in_contact[leg]

    def set_swing_reference(self, leg, position, velocity, acceleration, yaw=0.0):
        """Command the swing foot, flat on the ground and facing the heading."""
        pose = self.model.getPose(self.foot_frames[leg])
        pose.translation = tuple(position)
        # Level, and yawed with the body. Holding the foot world-aligned while
        # the robot turns lands it twisted against the heading, which both
        # fights the turn and feeds a yaw disturbance back into the trunk.
        pose.linear = _rot_z(yaw)
        self.swing_tasks[leg].setReference(
            pose,
            np.concatenate([velocity, np.zeros(3)]),
            np.concatenate([acceleration, np.zeros(3)]))

    # -------------------------------------------------------------- reference

    def set_reference(self, state, forces):
        """Track one MPC step.

        `state` is the 13-element MPC state and `forces` the (4, 3) ground
        reaction forces for the current step, both in the world frame.
        """
        roll, pitch, yaw = state[0:3]
        com_position = state[3:6]
        angular_velocity = state[6:9]
        linear_velocity = state[9:12]
        forces = np.asarray(forces, dtype=float).reshape(4, 3)

        # --- base attitude ---------------------------------------------------
        pose = self.model.getPose("pelvis")
        pose.linear = tf_trans.euler_matrix(roll, pitch, yaw)[0:3, 0:3]

        # Angular acceleration the MPC's forces actually produce about the CoM:
        # Iw^-1 * sum_i (r_i x f_i). Using the angular velocity in place of the
        # force here, as an earlier version did, is not a torque at all.
        inertia_world = _rot_z(yaw) @ self.inertia_body @ _rot_z(yaw).T
        moment = np.zeros(3)
        for i, frame in enumerate(self.contact_frames):
            r = self.model.getPose(frame).translation - com_position
            moment += np.cross(r, forces[i])
        angular_acceleration = np.linalg.solve(inertia_world, moment)

        self.base.setReference(
            pose,
            np.concatenate([np.zeros(3), angular_velocity]),
            np.concatenate([np.zeros(3), angular_acceleration]))

        # --- CoM --------------------------------------------------------------
        # Sum over contact points, so the reshape is (4, 3) summed down the
        # rows. Summing a (3, 4) view mixes x, y and z of different feet.
        linear_acceleration = forces.sum(axis=0) / self.mass + self.gravity
        self.com.setReference(com_position, linear_velocity, linear_acceleration)

        # --- desired contact forces -------------------------------------------
        for i in range(len(self.contact_frames)):
            self.wrench_tasks[i].setReference(forces[i])

    # ------------------------------------------------------------------ solve

    def update_model(self, q, dq):
        self.model.setJointPosition(q)
        self.model.setJointVelocity(dq)
        self.model.update()

    def solve(self):
        self.stack.update()
        solution = self.solver.solve()
        self.ddq = self.variables.getVariable("qddot").getValue(solution)
        self.contact_forces = [
            np.asarray(self.variables.getVariable(f).getValue(solution)).reshape(3)
            for f in self.contact_frames
        ]
        return self.ddq

    def inverse_dynamics(self):
        """Joint torques for the solved acceleration and contact forces."""
        self.model.setJointAcceleration(self.ddq)
        self.model.update()
        tau = self.model.computeInverseDynamics()
        for i, frame in enumerate(self.contact_frames):
            jacobian = self.model.getJacobian(frame)
            tau = tau - jacobian[:3, :].T @ self.contact_forces[i]
        return tau

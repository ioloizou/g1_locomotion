from xbot2_interface import pyxbot2_interface as xbi

import numpy as np

from pyopensot.tasks.acceleration import CoM, Cartesian, Postural, AngularMomentum, DynamicFeasibility
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone
import pyopensot as pysot
from ttictoc import tic, toc

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped


class WholeBodyID:
    def __init__(self, urdf, dt, q_init, friction_coef=0.8):
        self.dt = dt
        self.friction_coef = friction_coef
        self.model = xbi.ModelInterface2(urdf)
        self.qmin, self.qmax = self.model.getJointLimits()
        self.dqmax = self.model.getVelocityLimits()

        # Initial Joint Configuration
        self.q = q_init
        self.dq = np.zeros(self.model.nv)

    def setupProblem(self):
        self.model.setJointPosition(self.q)
        self.model.setJointVelocity(self.dq)
        self.model.update()

        # Instantiate Variables: qddot and contact forces (3 per contact)

        variables_vec = dict()
        variables_vec["qddot"] = self.model.nv

        line_foot_contact_frames = [
            "left_foot_line_contact_lower",
            "left_foot_line_contact_upper",
            "right_foot_line_contact_lower",
            "right_foot_line_contact_upper",
        ]

        # Hands may be added in the future
        self.contact_frames = line_foot_contact_frames

        for contact_frame in self.contact_frames:
            variables_vec[contact_frame] = 3
        self.variables = pysot.OptvarHelper(variables_vec)

        # Set CoM tracking task
        self.com = CoM(self.model, self.variables.getVariable("qddot"))
        self.com.setLambda(1000.0, 600.0)
        # FK at initial config
        com_ref, vel_ref, acc_ref = self.com.getReference()
        self.com0 = com_ref.copy()

        # Set the whole Cartesian task but later only orientation will be used
        base = Cartesian(
            "base", self.model, "world", "pelvis", self.variables.getVariable("qddot")
        )
        base.setLambda(100.0, 70.0)

        # Set the contact task
        contact_tasks = list()
        cartesian_contact_task_frames = [
            "left_foot_point_contact",
            "right_foot_point_contact",
        ]
        for cartesian_contact_task_frame in cartesian_contact_task_frames:
            contact_tasks.append(
                Cartesian(
                    cartesian_contact_task_frame,
                    self.model,
                    cartesian_contact_task_frame,
                    "world",
                    self.variables.getVariable("qddot"),
                )
            )

        posture = Postural(self.model, self.variables.getVariable("qddot"))
        posture.setLambda(300.0, 20.0)

        # Set the Angular momentum task
        angular_momentum = AngularMomentum(
            self.model, self.variables.getVariable("qddot")
        )

        # For the base task taking only the orientation part
        self.stack = 0.1 * self.com + 0.1 * (base % [3, 4, 5]) +0.1*angular_momentum + 0.1 * posture

        for i in range(len(cartesian_contact_task_frames)):
            self.stack = self.stack + 10.0 * (contact_tasks[i])

        force_variables = list()
        for i in range(len(self.contact_frames)):
            force_variables.append(self.variables.getVariable(self.contact_frames[i]))

        # Creates the stack.
        # Notice:  we do not need to keep track of the DynamicFeasibility constraint so it is created when added into the stack.
        # The same can be done with other constraints such as Joint Limits and Velocity Limits
        self.stack = (pysot.AutoStack(self.stack)) << DynamicFeasibility(
            "floating_base_dynamics",
            self.model,
            self.variables.getVariable("qddot"),
            force_variables,
            self.contact_frames,
        )
        self.stack = self.stack << JointLimits(
            self.model,
            self.variables.getVariable("qddot"),
            self.qmax,
            self.qmin,
            10.0 * self.dqmax,
            self.dt,
        )
        self.stack = self.stack << VelocityLimits(
            self.model, self.variables.getVariable("qddot"), self.dqmax, self.dt
        )
        for i in range(len(self.contact_frames)):
            T = self.model.getPose(self.contact_frames[i])
            mu = (T.linear, self.friction_coef)  # rotation is world to contact
            self.stack = self.stack << FrictionCone(
                self.contact_frames[i],
                self.variables.getVariable(self.contact_frames[i]),
                self.model,
                mu,
            )

        # Creates the solver
        self.solver = pysot.iHQP(self.stack)

    def updateModel(self, q, dq):
        self.model.setJointPosition(q)
        self.model.setJointVelocity(dq)
        self.model.update()

    def setReference(self, t):
        # alpha = 0.2
        com_ref = self.com0 
        # + alpha * np.array(
            # [0.0, np.sin(3.1415 * t), np.cos(3.1415 * t)]
        # )
        self.com.setReference(com_ref)

    def solveQP(self):
        self.x = self.solver.solve()
        self.ddq = self.variables.getVariable("qddot").getValue(self.x)
        self.contact_forces = list()
        for i in range(len(self.contact_frames)):
            self.contact_forces.append(
                self.variables.getVariable(self.contact_frames[i]).getValue(self.x)
            )

    def stepProblem(self, q_curr, dq_curr, t):
        self.updateModel(q_curr, dq_curr)
        self.setReference(t)
        self.stack.update()
        self.solveQP()
        return self.ddq, self.contact_forces

    def getInverseDynamics(self):
        # Update joint position
        self.model.setJointAcceleration(self.ddq)
        self.model.update()
        tau = self.model.computeInverseDynamics()
        for i in range(len(self.contact_frames)):
            Jc = self.model.getJacobian(self.contact_frames[i])
            tau = tau - Jc[:3, :].T @ np.array(self.contact_forces[i])
        return tau
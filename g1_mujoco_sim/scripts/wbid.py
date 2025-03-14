from xbot2_interface import pyxbot2_interface as xbi

import numpy as np

from pyopensot import GenericTask
from pyopensot.tasks.acceleration import CoM, Cartesian, Postural, AngularMomentum, DynamicFeasibility
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone, WrenchLimits
import pyopensot as pysot
from ttictoc import tic, toc

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped
from pyopensot import solver_back_ends

from matplotlib import pyplot as plt


def min_forces_task(model):
    A = np.zeros((2*6, model.nv + 2*6))
    b = np.zeros(2*6)

    A[:, model.nv:] = np.eye(2*6)
    return GenericTask("min_forces", A, b)



class WholeBodyID:
    def __init__(self, urdf, dt, q_init, friction_coef=1.):
        self.dt = dt
        self.friction_coef = friction_coef
        self.model = xbi.ModelInterface2(urdf)
        self.qmin, self.qmax = self.model.getJointLimits()
        self.dqmax = self.model.getVelocityLimits()

        # Initial Joint Configuration
        self.q = q_init
        self.dq = np.zeros(self.model.nv)

        # ----- Plot Initialization for CoM Trajectory -----
        self.fig, self.ax = plt.subplots()
        # Initialize empty lines for desired and current CoM (using red and blue colors)
        self.desired_com_line, = self.ax.plot([], [], 'r-', label='Desired CoM')
        self.current_com_line, = self.ax.plot([], [], 'b-', label='Current CoM')
        self.ax.set_xlabel('X Position [m]')
        self.ax.set_ylabel('Y Position [m]')
        self.ax.set_title('Center of Mass Trajectory')
        self.ax.legend()
        self.ax.grid(True)
        plt.ion()    # Turn on interactive mode so that the plot can update live
        plt.show()
        # Prepare lists to store the trajectory data
        self.desired_com_x = []
        self.desired_com_y = []
        self.current_com_x = []
        self.current_com_y = []
        # -----------------------------------------------------

    def setupProblem(self):
        self.model.setJointPosition(self.q)
        self.model.setJointVelocity(self.dq)
        self.model.update()

        # Instantiate Variables: qddot and contact forces (3 per contact)

        variables_vec = dict()
        variables_vec["qddot"] = self.model.nv

        line_foot_contact_frames = [
            "left_foot_upper_left",
            "left_foot_upper_right",
            "left_foot_lower_left",
            "left_foot_lower_right",
            "right_foot_upper_left",
            "right_foot_upper_right",
            "right_foot_lower_left",
            "right_foot_lower_right",
        ]

        cartesian_contact_task_frames = [
            "left_foot_point_contact",
            "right_foot_point_contact",
        ]

        self.contact_frames = cartesian_contact_task_frames

        # Hands may be added in the future
        #self.contact_frames = line_foot_contact_frames

        for contact_frame in self.contact_frames:
            variables_vec[contact_frame] = 6
        self.variables = pysot.OptvarHelper(variables_vec)

        # Set CoM tracking task
        self.com = CoM(self.model, self.variables.getVariable("qddot"))

        # FK at initial config
        com_ref, vel_ref, acc_ref = self.com.getReference()
        self.com0 = com_ref.copy()

        # Set the whole Cartesian task but later only orientation will be used
        base = Cartesian(
            "base", self.model, "world", "pelvis", self.variables.getVariable("qddot")
        )

        # Set the contact task
        self.contact_tasks = list()
        

        for cartesian_contact_task_frame in cartesian_contact_task_frames:
            self.contact_tasks.append(
                Cartesian(
                    cartesian_contact_task_frame,
                    self.model,
                    cartesian_contact_task_frame,
                    "world",
                    self.variables.getVariable("qddot"),
                )
            )

        """self.wrench_limits = list()
        for contact_frame in line_foot_contact_frames:
            self.wrench_limits.append(
                WrenchLimits(
                    contact_frame,
                    np.array([-1000., -1000., -1000.]),
                    np.array([1000., 1000., 1000.]),
                    self.variables.getVariable(contact_frame))
            )"""

        posture = Postural(self.model, self.variables.getVariable("qddot"))

        # Set the Angular momentum task
        angular_momentum = AngularMomentum(
            self.model, self.variables.getVariable("qddot")
        )

        # For the base task taking only the orientation part
        self.stack = 1. * self.com + 1. * (base % [3, 4, 5]) + 1. * posture % list(range(18, self.model.nv)) + 1e-6 * min_forces_task(self.model)
        for i in range(len(cartesian_contact_task_frames)):
            self.stack = self.stack + 1.0 * (self.contact_tasks[i])

        #self.stack = self.stack / posture

        force_variables = list()
        for i in range(len(self.contact_frames)):
            force_variables.append(self.variables.getVariable(self.contact_frames[i]))

        # Gains
        #self.com.setLambda(1000.0, 600.0)
        #base.setLambda(100.0, 70.0)
        #for contact_task in self.contact_tasks:
        #    contact_task.setLambda(100.0, 10.0)
        #posture.setLambda(300.0, 20.0)
        #angular_momentum.setLambda(100.0)

        self.com.setLambda(250.0, 10.)
        base.setLambda(250.0, 10.)
        for contact_task in self.contact_tasks:
            contact_task.setLambda(250.0, 10.)
        posture.setLambda(100.0, 10.)
        angular_momentum.setLambda(0.0)

        # Creates the stack.
        # Notice:  we do not need to keep track of the DynamicFeasibility constraint so it is created when added into the stack.
        # The same can be done with other constraints such as Joint Limits and Velocity Limits
        self.stack = (pysot.AutoStack(self.stack)) << DynamicFeasibility(
        #self.stack << DynamicFeasibility(
            "floating_base_dynamics",
            self.model,
            self.variables.getVariable("qddot"),
            force_variables,
            self.contact_frames,
        )

        #for i in range(len(cartesian_contact_task_frames)):
        #    self.stack << self.contact_tasks[i]


        """self.stack = self.stack << JointLimits(
            self.model,
            self.variables.getVariable("qddot"),
            self.qmax,
            self.qmin,
            10.0 * self.dqmax,
            self.dt,
        )"""
        """self.stack = self.stack << VelocityLimits(
            self.model, self.variables.getVariable("qddot"), self.dqmax, self.dt
        )"""
        for i in range(len(self.contact_frames)):
            T = self.model.getPose(self.contact_frames[i])
            mu = (T.linear, self.friction_coef)  # rotation is world to contact
            self.stack = self.stack << FrictionCone(
                self.contact_frames[i],
                self.variables.getVariable(self.contact_frames[i]),
                self.model,
                mu,
            )# << self.wrench_limits[i]

        # Creates the solver
        self.solver = pysot.iHQP(self.stack, eps_regularisation=1e-36) #, be_solver=solver_back_ends.eiQuadProg)
        #self.solver = pysot.nHQP(self.stack.getStack(), self.stack.getBounds(), 1e-6)

    def updateModel(self, q, dq):
        self.model.setJointPosition(q)
        self.model.setJointVelocity(dq)
        self.model.update()

    def update_plot(self):
        desired_com, _, _ = self.com.getReference()
        current_com = self.com.getActualPose()
        
        # Append the new data (only x and y are plotted)
        self.desired_com_x.append(desired_com[0])
        self.desired_com_y.append(desired_com[1])
        self.current_com_x.append(current_com[0])
        self.current_com_y.append(current_com[1])

        # Update the data of the plot lines
        #self.desired_com_line.set_data(self.desired_com_x, self.desired_com_y)
        #self.current_com_line.set_data(self.current_com_x, self.current_com_y)
        self.desired_com_line.set_data(range(len(self.desired_com_y)), self.desired_com_y)
        self.current_com_line.set_data(range(len(self.current_com_y)), self.current_com_y)

        # Rescale the axis to show the full trajectory
        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

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
        self.stack.update()
        self.solveQP()
        self.update_plot()
        return self.ddq, self.contact_forces

    def getInverseDynamics(self):
        # Update joint position
        self.model.setJointAcceleration(self.ddq)
        self.model.update()
        tau = self.model.computeInverseDynamics()
        for i in range(len(self.contact_frames)):
            Jc = self.model.getJacobian(self.contact_frames[i])
            tau = tau - Jc[:6, :].T @ np.array(self.contact_forces[i])
        return tau
from xbot2_interface import pyxbot2_interface as xbi

import numpy as np

from pyopensot.tasks.acceleration import CoM, Cartesian, Postural, AngularMomentum, DynamicFeasibility
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits, TorqueLimits
from pyopensot.constraints.force import FrictionCone, WrenchLimits
from pyopensot.tasks import MinimizeVariable
from pyopensot.variables import Torque
from pyopensot import solver_back_ends
import pyopensot as pysot
import tf.transformations as tf_trans
from ttictoc import tic, toc


class WholeBodyID:
    def __init__(self, urdf, dt, q_init, friction_coef=0.8):
        self.dt = dt
        self.friction_coef = friction_coef
        self.model = xbi.ModelInterface2(urdf)
        self.qmin, self.qmax = self.model.getJointLimits()
        self.dqmax = self.model.getVelocityLimits()
        self.torque_limits = self.model.getEffortLimits()

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
        com_gain = 4.
        com_Kp = np.eye(3) * 100. * com_gain
        self.com.setKp(com_Kp)
        com_Kd = np.diag([30., 30., 50.]) * com_gain
        self.com.setKd(com_Kd)
        self.com.setLambda(1.0, 1.0)

        # FK at initial config
        self.com_ref, vel_ref, acc_ref = self.com.getReference()
        self.com0 = self.com_ref.copy()

        # Set the whole Cartesian task but later only orientation will be used
        self.base = Cartesian(
            "base", self.model, "world", "pelvis", self.variables.getVariable("qddot")
        )
        base_gain = 12.
        base_Kp = np.diag([1., 1., 1., 10., 10., 20.]) * base_gain
        self.base.setKp(base_Kp)
        base_Kd = np.diag([1., 1., 1., 50., 50., 50.]) * base_gain
        self.base.setKd(base_Kd)
        self.base.setLambda(1.0, 1.0)

        # Set the contact task
        self.contact_tasks = list()
        self.cartesian_contact_task_frames = [
            "left_foot_point_contact",
            "right_foot_point_contact",
        ]
        for cartesian_contact_task_frame in self.cartesian_contact_task_frames:
            self.contact_tasks.append(
                Cartesian(
                    cartesian_contact_task_frame,
                    self.model,
                    cartesian_contact_task_frame,
                    "world",
                    self.variables.getVariable("qddot"),
                )
            )

        # Set swing task (Dummy hands now)
        self.swing_tasks = list()
        self.swing_cartesian_frames =  [
            "left_foot_point_contact",
            "right_foot_point_contact",
        ]
        for swing_cartesian_frame in self.swing_cartesian_frames:
            self.swing_tasks.append(
                Cartesian(
                    swing_cartesian_frame,
                    self.model,
                    swing_cartesian_frame,
                    "world",
                    self.variables.getVariable("qddot"),
                )
            )
            self.swing_tasks[-1].setLambda(1., 1.)
            swing_gain = 4.
            swing_Kp = np.diag([550., 750., 560., 70., 70., 70.]) * swing_gain
            swing_Kd = np.diag([10., 10., 17., 7., 7., 7.]) * swing_gain
            self.swing_tasks[-1].setKp(swing_Kp)
            self.swing_tasks[-1].setKd(swing_Kd)
        
        # Set wrench limits (x, y are irrelevant since are constrained by friction cone)
        self.wrench_limits = list()
        for contact_frame in line_foot_contact_frames:
            self.wrench_limits.append(
                WrenchLimits(
                    contact_frame,
                    np.array([-1000., -1000., 10.]),
                    np.array([1000., 1000., 1000.]),
                    self.variables.getVariable(contact_frame))
            )

        posture_gain = 200.
        posture = Postural(self.model, self.variables.getVariable("qddot"))
        posture_Kp = np.eye(self.model.nv) * 2. * posture_gain
        posture.setKp(posture_Kp)
        posture_Kd = np.diag([0.2] * self.model.nv) * posture_gain
        posture.setKd(posture_Kd)
        posture.setLambda(1.0, 1.0)   

        # Set the Angular momentum task
        angular_momentum = AngularMomentum(
            self.model, self.variables.getVariable("qddot")
        )
        angular_momentum_Kd = np.diag([3.] * 3)
        angular_momentum.setMomentumGain(angular_momentum_Kd)
        angular_momentum.setLambda(1.)

        force_variables = list()
        for i in range(len(self.contact_frames)):
            force_variables.append(self.variables.getVariable(self.contact_frames[i]))

        # Torque 
        torques = Torque(self.model, self.variables.getVariable("qddot"), self.contact_frames, force_variables)

        # Regularization of acceleration
        req_qddot = MinimizeVariable("acceleration", self.variables.getVariable("qddot"))

        # # Regularization of forces
        req_forces_0 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[0]))
        req_forces_1 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[1]))
        req_forces_2 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[2]))
        req_forces_3 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[3]))

        min_force_weight = 1e-5
        # For the self.base task taking only the orientation part
        
        #for joint_name in self.model.getJointNames():
        #    print(f"{joint_name} : {self.model.getJointId(joint_name)}")
        #exit()

        self.stack = 3.5*self.com + 0.4*(posture%[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])+ 0.005*req_qddot 
        self.stack += 0.5*angular_momentum
        # self.stack += min_force_weight*req_forces_0 + min_force_weight*req_forces_1 + min_force_weight*req_forces_2 + min_force_weight*req_forces_3
        self.stack += 1e-8 * MinimizeVariable("min_torques", torques)
        # 
        self.stack += 3.5*(self.base%[3, 4 ,5])
        
        for i in range(len(self.cartesian_contact_task_frames)):
            self.contact_tasks[i].setLambda(300.0, 20.)
            self.stack = self.stack + 4.5 * (self.contact_tasks[i]) + 2.9*self.swing_tasks[i]

        # Task for factual - fdesired
        self.wrench_tasks = list()
        for contact_frame in self.contact_frames:
            self.wrench_tasks.append(MinimizeVariable(contact_frame, self.variables.getVariable(contact_frame)))
            self.stack = self.stack + 0.05*(self.wrench_tasks[-1])
        
        self.dynamics_constraint = DynamicFeasibility(
            "floating_base_dynamics",
            self.model,
            self.variables.getVariable("qddot"),
            force_variables,
            self.contact_frames,
        )

        # Creates the stack.
        # Notice:  we do not need to keep track of the DynamicFeasibility constraint so it is created when added into the stack.
        # The same can be done with other constraints such as Joint Limits and Velocity Limits
        self.stack = (pysot.AutoStack(self.stack)) << self.dynamics_constraint
        self.stack = self.stack << JointLimits(
            self.model,
            self.variables.getVariable("qddot"),
            self.qmax,
            self.qmin,
            10.0 * self.dqmax,
            self.dt)
        self.stack = self.stack << VelocityLimits(
            self.model, self.variables.getVariable("qddot"), self.dqmax, self.dt)

        self.stack = self.stack << TorqueLimits(
             self.model,
             self.variables.getVariable("qddot"),
             force_variables,
             self.contact_frames,
             self.torque_limits)
          
        for i in range(len(self.contact_frames)):
            R = np.eye(3)
            mu = (R, self.friction_coef)  # rotation is world to contact
            self.stack = self.stack << FrictionCone(
                self.contact_frames[i],
                self.variables.getVariable(self.contact_frames[i]),
                self.model,
                mu) << self.wrench_limits[i]
            

        # Creates the solver
        self.solver = pysot.iHQP(self.stack) #, solver_back_ends.eiQuadProg)
                                 

    def updateModel(self, q, dq):
        self.model.setJointPosition(q)
        self.model.setJointVelocity(dq)
        self.model.update()

    def setReference(self, t, x_opt1=None, u_opt0=None, foot_positions_curr=None):
        if x_opt1 is None:
            alpha = 0.04
            self.com_ref[0] = self.com0[0] 
            self.com_ref[1] = self.com0[1] #+ alpha * np.sin(3.1415 * t) 
            self.com_ref[2] = self.com0[2] #+ alpha * np.cos(2* 3.1415 * t)
            print("t", t)
            print("com0", self.com0)
            print("self.com_ref", self.com_ref)
            self.com.setReference(self.com_ref)
        else:
            # The base reference is the first 6 elements of x_opt1 but the orientation is only used
            # for the base task

            # x_opt1[0:3] contains roll, pitch, yaw for the torso orientation.
            roll, pitch, yaw = x_opt1[0:3]
            R = tf_trans.euler_matrix(roll, pitch, yaw)[0:3, 0:3]

            # # Get current full homogeneous transformation. 
            base_affine = self.base.getReference() 

            # # Update only the rotation part.
            base_affine[0].linear = R

            # # Linear and Angular Velocity. Linear drops in the stack
            linear_velocity = x_opt1[9:12]
            angular_velocity = x_opt1[6:9]
            velocity = np.hstack((linear_velocity, angular_velocity))
            
            # Since i use in MPC the torso inertia, 
            self.inertia_torso = np.array([
                [8.20564e-2, 0., 0.],
                [0., 8.05015e-2, 0.],
                [0., 0., 0.32353e-2]
            ])

            # If i turn i need to rotate it based on yaw
            inertia_torso_inv = np.linalg.inv(self.inertia_torso)
            
            r = np.array(foot_positions_curr) - np.tile(x_opt1[3:6], (4,1))
            
            sum_r_cross_omega = np.zeros((1, 3))
            for i in range(4):
                sum_r_cross_omega = sum_r_cross_omega + np.cross(r[i, :], x_opt1[6:9])

            # The linear is just to satisfy opensot since is omitted in the stack
            linear_acceleration_reference = np.zeros((3,1))
            angular_acceleration_reference = inertia_torso_inv @ sum_r_cross_omega.T
            # angular_acceleration_reference = np.zeros((3,1))

            acceleration_reference = np.vstack((linear_acceleration_reference, angular_acceleration_reference)) 
            # Pass the updated homogeneous transformation.
            self.base.setReference(base_affine[0], velocity, acceleration_reference)
            
            # Calculate from MPC result
            gravity = np.array([0, 0, -9.80665])

            # The ending vector is 3,1 and all the elements are summed with every third element from that element from u_opt0
            sum_forces = np.sum(np.reshape(u_opt0, (3, 4)), axis=1)
            linear_acceleration_reference = sum_forces/self.model.getMass() + gravity

            # linear_acceleration_reference = np.zeros(3)
            self.com.setReference(x_opt1[3:6], x_opt1[9:12], linear_acceleration_reference)

            for i in range(len(self.contact_frames)):
                self.wrench_tasks[i].setReference(u_opt0[i*3:i*3+3])


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
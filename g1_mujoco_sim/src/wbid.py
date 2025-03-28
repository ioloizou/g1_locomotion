from xbot2_interface import pyxbot2_interface as xbi

import numpy as np

from pyopensot.tasks.acceleration import CoM, Cartesian, Postural, AngularMomentum, DynamicFeasibility
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone, WrenchLimits
from pyopensot import solver_back_ends
import pyopensot as pysot
import tf.transformations as tf_trans
from ttictoc import tic, toc

def MinimizeVariable(name, opt_var): 
	'''Task to regularize a variable using a generic task'''
	A = opt_var.getM()

	# The minus because y = Mx + q is mapped on ||Ax - b|| 
	b = -opt_var.getq()
	task = pysot.GenericTask(name, A, b, opt_var)


	# Setting the regularization weight.
	task.setWeight(1.)
	task.update()

	return task

def Wrench(name, distal_link, base_link, wrench):
	'''Task to minimize f-fd using a generic task'''

	# print(wrench)
	# print(wrench.getM().shape)
	# print(wrench.getq())

	A = np.eye(3)
	b =	- wrench.getq()
	
	return pysot.GenericTask(name, A, b, wrench) 

def setDesiredForce(Wrench_task, wrench_desired, wrench):
	# b = -(wrench - wrench_desired).getq()

	b = wrench_desired - wrench.getq()
	
	# print(f"b: {b}")
	
	# print(f'wrench_desired_dimensions: {wrench_desired.shape}')
	# print(f'wrench: {wrench}')
	# print(wrench.getM().shape)
	# print(f'wrench_dimensions: {wrench.getq().shape}')
	# print(f'b_dimensions: {b.shape}')
	Wrench_task.setb(b)

class WholeBodyID:
    def __init__(self, urdf, dt, q_init, friction_coef=0.3):
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
        com_gain = 1.
        com_Kp = np.eye(3) * 100. * com_gain
        self.com.setKp(com_Kp)
        com_Kd = np.diag([30., 30., 50.]) * com_gain
        self.com.setKd(com_Kd)
        self.com.setLambda(1.0, 1.0)

        # self.com.setKd(50)
        # self.com.setLambda(1.0, 1.0)
        # FK at initial config
        self.com_ref, vel_ref, acc_ref = self.com.getReference()
        self.com0 = self.com_ref.copy()

        # Set the whole Cartesian task but later only orientation will be used
        self.base = Cartesian(
            "base", self.model, "world", "pelvis", self.variables.getVariable("qddot")
        )
        base_gain = 1.
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
        self.swing_cartesian_frames = [
            "left_hand_point_contact",
            "right_hand_point_contact",
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
            swing_gain = 1.
            swing_Kp = np.diag([350., 350., 560., 70., 70., 70.]) * swing_gain
            swing_Kd = np.diag([10., 10., 17., 7., 7., 7.]) * swing_gain
            self.swing_tasks[-1].setKp(swing_Kp)
            self.swing_tasks[-1].setKd(swing_Kd)
        
        # Set wrench limits (x, y are irrelevant since are constrained by friction cone)
        self.wrench_limits = list()
        for contact_frame in line_foot_contact_frames:
            self.wrench_limits.append(
                WrenchLimits(
                    contact_frame,
                    np.array([0., 0., 3.]),
                    np.array([666., 666., 666.]),
                    self.variables.getVariable(contact_frame))
            )

        posture_gain = 40.
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
        
        

        # Regularization of acceleration
        req_qddot = MinimizeVariable("acceleration", self.variables.getVariable("qddot"))

        # # Regularization of forces
        req_forces_0 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[0]))
        req_forces_1 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[1]))
        req_forces_2 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[2]))
        req_forces_3 = MinimizeVariable("min_forces", self.variables.getVariable(self.contact_frames[3]))

        min_force_weight = 1e-5
        # For the self.base task taking only the orientation part
        self.stack = 1.0*self.com + 0.02*(posture%[18, 19, 20, 21, 22, 23]) + 0.3*angular_momentum + 0.005*req_qddot + min_force_weight*req_forces_0 + min_force_weight*req_forces_1 + min_force_weight*req_forces_2 + min_force_weight*req_forces_3
        # , 24, 25, 26, 27, 28
        # self.stack += 1.0*(self.base%[3, 4 ,5])
        
        for i in range(len(self.cartesian_contact_task_frames)):
            self.contact_tasks[i].setLambda(500.0, 20.)
            self.stack = self.stack + 10.0 * (self.contact_tasks[i])

        # Task for factual - fdesired
        self.wrench_tasks = list()
        for contact_frame in self.contact_frames:
            self.wrench_tasks.append(Wrench(contact_frame, contact_frame, "pelvis", self.variables.getVariable(contact_frame)))
            self.stack = self.stack + 0.1*(self.wrench_tasks[-1])
        
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
            ) << self.wrench_limits[i]

        # Creates the solver
        self.solver = pysot.iHQP(self.stack) #, solver_back_ends.eiQuadProg)
                                 

    def updateModel(self, q, dq):
        self.model.setJointPosition(q)
        self.model.setJointVelocity(dq)
        self.model.update()

    def setReference(self, t, x_opt1=None, u_opt0=None):
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
            
            # # Pass the updated homogeneous transformation.
            self.base.setReference(base_affine[0], velocity)
            # CoM position, CoM velocity, CoM acceleration references 
            
            # Calculate from MPC result
            gravity = np.array([0, 0, -9.80665])

            # The ending vector is 3,1 and all the elements are summed with every third element from that element from u_opt0
            sum_forces = np.sum(np.reshape(u_opt0, (3, 4)), axis=1)
            acceleration_reference = sum_forces/self.model.getMass() + gravity

            # acceleration_reference = np.zeros(3)
            self.com.setReference(x_opt1[3:6], x_opt1[9:12], acceleration_reference)

            for i in range(len(self.contact_frames)):
                setDesiredForce(self.wrench_tasks[i], u_opt0[i*3:i*3+3], self.variables.getVariable(self.contact_frames[i]))            
         
        


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
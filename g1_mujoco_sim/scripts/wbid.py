from xbot2_interface import pyxbot2_interface as xbi

import numpy as np

from pyopensot.tasks.acceleration import CoM, Cartesian, Postural, AngularMomentum, DynamicFeasibility
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone
import pyopensot as pysot
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
	
	print(f"b: {b}")
	
	# print(f'wrench_desired_dimensions: {wrench_desired.shape}')
	# print(f'wrench: {wrench}')
	# print(wrench.getM().shape)
	# print(f'wrench_dimensions: {wrench.getq().shape}')
	# print(f'b_dimensions: {b.shape}')
	Wrench_task.setb(b)

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
        base = Cartesian(
            "base", self.model, "world", "pelvis", self.variables.getVariable("qddot")
        )
        base_gain = 1.
        base_Kp = np.diag([1., 1., 1., 10., 10., 20.]) * base_gain
        base.setKp(base_Kp)
        base_Kd = np.diag([1., 1., 1., 50., 50., 50.]) * base_gain
        base.setKd(base_Kd)
        base.setLambda(1.0, 1.0)

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
        posture_gain = 10.
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

        # Regularization of acceleration
        req_qddot = MinimizeVariable("acceleration", self.variables.getVariable("qddot"))

        # For the base task taking only the orientation part
        self.stack = 1.0*self.com + 1.0*(base%[3, 4, 5]) + 0.02*(posture%[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]) + 0.3*angular_momentum + 0.005*req_qddot

        for i in range(len(cartesian_contact_task_frames)):
            self.stack = self.stack + 10.0 * (contact_tasks[i])

        # Task for factual - fdesired
        self.wrench_tasks = list()
        for contact_frame in self.contact_frames:
            self.wrench_tasks.append(Wrench(contact_frame, contact_frame, "pelvis", self.variables.getVariable(contact_frame)))
            self.stack = self.stack + 0.01*(self.wrench_tasks[-1])
        
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

    def setReference(self, t, com_opt1=None, u_opt0=None):
        if com_opt1 is None:
            alpha = 0.04
            self.com_ref[0] = self.com0[0] 
            self.com_ref[1] = self.com0[1] #+ alpha * np.sin(3.1415 * t) 
            self.com_ref[2] = self.com0[2] #+ alpha * np.cos(2* 3.1415 * t)
            print("t", t)
            print("com0", self.com0)
            print("self.com_ref", self.com_ref)
            self.com.setReference(self.com_ref)
        else:
            self.com.setReference(com_opt1)

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
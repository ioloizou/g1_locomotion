#!/usr/bin/env python3

import os
import time
import mujoco
import mujoco_viewer
import rospkg
import rospy
import numpy as np

from config import q_init
from PD_controller import PD_controller


from ttictoc import tic, toc
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped

from xbot2_interface import pyxbot2_interface as xbi
from pyopensot.tasks.acceleration import CoM, Cartesian, Postural, DynamicFeasibility, AngularMomentum
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone
import pyopensot as pysot

def MinimizeVariable(name, opt_var): 
	'''Task to regularize a variable using a generic task'''
	A = opt_var.getM()

	# The minus because y = Mx + q is mapped on ||Ax - b|| 
	b = -opt_var.getq()
	task = pysot.GenericTask(name, A, b, opt_var)


	# Setting the regularization weight.
	task.setWeight(1.)
	task.update()

	# print(f"MinVar A:\n {task.getA()}")
	# print(f"MinVar b:\n {task.getb()}")
	# print(f"MinVar W:\n {task.getWeight()}")

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

class G1MujocoSimulation:
    def __init__(self):
        rospy.init_node("g1_mujoco_sim", anonymous=True)

        # Find the model path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("g1_mujoco_sim")
        model_path = os.path.join(pkg_path, "..", "g1_description", "g1_23dof.xml")
        
        # Load the URDF
        with open(os.path.join(pkg_path, "..", "g1_description", "g1_23dof.urdf"), 'r') as urdf_file:
            self.urdf = urdf_file.read()

        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        print("self.data.qpos ===========================\n", self.data.qpos)

        self.data.qpos = q_init

        # Real-time settings
        self.sim_timestep = self.model.opt.timestep
        # print("self.sim_timestep ===========================\n", self.sim_timestep)
        self.real_time_factor = 1.  # 1.0 = real time

        # Create viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Running flag
        self.running = True

    def permute_muj_to_xbi(self, xbi_qpos):
        """Convert mujoco qpos to xbi qpos."""
        # Place the 4th element at the 7th position
        xbi_qpos_perm = xbi_qpos.copy()
        xbi_qpos_perm[6] = xbi_qpos[3]
        xbi_qpos_perm[3] = xbi_qpos[6]
        return xbi_qpos_perm

    def setup_WBID(self):
        
        # Get model from urdf
        self.xbi_model = xbi.ModelInterface2(self.urdf)
        qmin, qmax = self.xbi_model.getJointLimits()
        dqmax = self.xbi_model.getVelocityLimits()
        # Initial Joint Configuration
        self.q = q_init.copy()
        self.dq = np.zeros(self.xbi_model.nv)
        self.xbi_model.setJointPosition(self.permute_muj_to_xbi(self.q))
        print("self.xbi_model.getJointPosition() ===========================", self.xbi_model.getJointPosition())
        self.xbi_model.setJointVelocity(self.dq)
        print("self.xbi_model.getJointVelocity() ===========================", self.xbi_model.getJointVelocity())
        self.xbi_model.update()
        # Instantiate Variables: qddot and contact forces (3 per contact)
        
        variables_vec = dict()
        variables_vec["qddot"] = self.xbi_model.nv

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
        self.com = CoM(self.xbi_model, self.variables.getVariable("qddot"))
        self.com.setLambda(1000., 600.)
        # FK at initial config
        self.com_ref, self.vel_ref, self.acc_ref = self.com.getReference()
        self.com0 = self.com_ref.copy()

        # Set the whole Cartesian task but later only orientation will be used
        base = Cartesian(
            "base", self.xbi_model, "world", "pelvis", self.variables.getVariable("qddot")
        )
        base.setLambda(100., 70.)

        # Set the Angular momentum task
        angular_momentum = AngularMomentum(self.xbi_model, self.variables.getVariable("qddot"))
        # angular_momentum.setLambda(100.)
        
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
                    self.xbi_model,
                    cartesian_contact_task_frame,
                    "world",
                    self.variables.getVariable("qddot"),
                )
            )
            # contact_tasks[-1].setLambda(100., 40.)
            

        posture = Postural(self.xbi_model, self.variables.getVariable("qddot"))
        posture.setLambda(300., 20.)
        
        # Regularization task
        reg_qddot = MinimizeVariable("req_qddot", self.variables.getVariable("qddot"))
        

        # For the base task taking only the orientation part
        self.stack = 0.1*self.com + 0.1*(base % [3, 4, 5]) + 0.1*angular_momentum + 0.1*posture
      
        self.wrench_tasks = list()
        for contact_frame in self.contact_frames:
            self.wrench_tasks.append(Wrench(contact_frame, contact_frame, "pelvis", self.variables.getVariable(contact_frame)))
            setDesiredForce(self.wrench_tasks[-1], [0, 0, 0.], self.variables.getVariable(contact_frame))
            # self.stack = self.stack + 0.000000001*(self.wrench_tasks[-1])

        for i in range(len(cartesian_contact_task_frames)):
            self.stack = self.stack + 15.*(contact_tasks[i])

        self.force_variables = list()
        for i in range(len(self.contact_frames)):
            self.force_variables.append(self.variables.getVariable(self.contact_frames[i]))

        # Creates the stack.
        self.stack = (pysot.AutoStack(self.stack)) << DynamicFeasibility(
            "floating_base_dynamics",
            self.xbi_model,
            self.variables.getVariable("qddot"),
            self.force_variables,
            self.contact_frames,
        )
        self.stack = self.stack << JointLimits(
            self.xbi_model,
            self.variables.getVariable("qddot"),
            qmax,
            qmin,
            10.0 * dqmax,
            self.sim_timestep,
        )
        self.stack = self.stack << VelocityLimits(
            self.xbi_model, self.variables.getVariable("qddot"), dqmax, self.sim_timestep
        )
        for i in range(len(self.contact_frames)):
            T = self.xbi_model.getPose(self.contact_frames[i])
            mu = (T.linear, 0.8)  # rotation is world to contact
            self.stack = self.stack << FrictionCone(
                self.contact_frames[i],
                self.variables.getVariable(self.contact_frames[i]),
                self.xbi_model,
                mu,
            )

        # Creates the solver
        self.solver = pysot.iHQP(self.stack)

        # Set amplitude for CoM task
        self.alpha = 0.4

    def sim_step(self):
        """Perform a single simulation step."""
        
        # print("data.qpos ===========================")
        # print(self.data.qpos)
        # print("data.qvel ===========================")
        # print(self.data.qvel)
        
        self.xbi_model.setJointPosition(self.permute_muj_to_xbi(self.data.qpos))
        # print('permuted qpos ===========================')
        # print(self.permute_muj_to_xbi(self.data.qpos))
        
        self.xbi_model.setJointVelocity(self.data.qvel)
        # print("self.xbi_model.getJointVelocity() ===========================")
        # print(self.xbi_model.getJointVelocity())
        
        self.xbi_model.update()
        
        # print("self.com0 ===========================")
        # print(self.com0)

        # Compute new reference for CoM task
        print("self.sim_time =========================== \n", self.sim_time)
        self.com_ref[0] = self.com0[0]
        self.com_ref[1] = self.com0[1] 
        # + self.alpha * np.cos(3.1415 * self.sim_time)
        self.com_ref[2] = self.com0[2]
        # + self.alpha * np.sin(3.1415 * self.sim_time)
        print("self.com_ref =========================== \n", self.com0)
        self.com.setReference(self.com_ref)

        for i in range(len(self.contact_frames)):
            setDesiredForce(self.wrench_tasks[i], [0, 0, 0], self.variables.getVariable(self.contact_frames[i]))

        # Update stack
        self.stack.update()
        
        # Solve
        x = self.solver.solve()
        ddq = self.variables.getVariable("qddot").getValue(x)


        # I NEED TO PASS THE CURRENT POSITION AND VEL
        self.q = self.xbi_model.sum(
            self.q, self.dq * self.sim_timestep + 0.5 * ddq * self.sim_timestep**2
        )

        # I NDEED TO PASS THE CURRENT VEL
        # Update joint velocity
        self.dq = self.dq + ddq * self.sim_timestep

        # Extract the force values
        forces = list()
        for i in range(len(self.contact_frames)):
            force_var = self.variables.getVariable(self.contact_frames[i])
            forces.append(force_var.getValue(x))
        
        print("forces ===========================")
        print(np.array(forces))
        
        # Update joint position
        self.xbi_model.setJointAcceleration(ddq)
        self.xbi_model.update()

        tau = self.xbi_model.computeInverseDynamics()
        
        # Reaction Forces to torques
        for i in range(len(self.contact_frames)):
            Jc = self.xbi_model.getJacobian(self.contact_frames[i])
            tau = tau - Jc[:3, :].T  @ np.array(forces[i])
        
        print("tau ===========================")
        print(tau)
        
        print("hi")
        # Exclude floating base
        self.data.ctrl = tau[6:]
                
        self.pass_count += 1
        # if self.pass_count >= 1:
            # exit()

        mujoco.mj_step(self.model, self.data)

    def run(self):
        """Run simple real-time simulation."""
        prev_time = time.time()
        self.sim_time = 0.0
        self.pass_count = 0

        try:
            while self.running and not rospy.is_shutdown() and self.viewer.is_alive:
                # Get real time elapsed since last step
                current_time = time.time()
                elapsed_wall_time = current_time - prev_time
                prev_time = current_time

                # Compute simulation time to advance (applying real-time factor)
                sim_time_to_advance = elapsed_wall_time * self.real_time_factor

                # Calculate steps needed
                steps = max(1, int(sim_time_to_advance / self.sim_timestep))

                # Step the simulation
                for _ in range(steps):
                    self.sim_step()
                    self.sim_time += self.sim_timestep

                # Render the current state
                self.viewer.render()

        except Exception as e:
            rospy.logerr(f"Simulation error: {e}")
        finally:
            if self.viewer:
                self.viewer.close()


if __name__ == "__main__":
    # Run simulation
    sim = G1MujocoSimulation()
    sim.setup_WBID()
    sim.run()

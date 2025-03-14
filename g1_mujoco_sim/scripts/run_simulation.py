#!/usr/bin/env python3

import os
import time
import mujoco
import mujoco_viewer
import rospkg
import rospy
import numpy as np
from ttictoc import tic, toc
import casadi as cs

from config import q_init
from wbid import WholeBodyID
from srbd_horizon import LIPMpc
from horizon.ros import utils as horizon_ros_utils
from scipy.spatial.transform import Rotation as R

def loadURDF(pkg_path):
    with open(
        os.path.join(pkg_path, "..", "g1_description", "g1_23dof.urdf"), "r"
    ) as urdf_file:
        urdf = urdf_file.read()
    return urdf

class G1MujocoSimulation:
    def __init__(self, q_init):
        rospy.init_node("g1_mujoco_sim", anonymous=True)

        # Find the model path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("g1_mujoco_sim")
        model_path = os.path.join(pkg_path, "..", "g1_description", "g1_23dof.xml")

        # Load the URDF for xbot
        self.urdf = loadURDF(pkg_path)
        horizon_ros_utils.roslaunch("srbd_horizon", "SRBD_g1_line_feet.launch")


        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Set initial joint configuration
        self.data.qpos = self.permute_muj_to_xbi(q_init)

        # Real-time settings
        self.sim_timestep = self.model.opt.timestep
        self.real_time_factor = 1.  # 1.0 = real time

        # Create viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Running flag
        self.running = True

        # Setup the whole body ID
        self.WBID = WholeBodyID(self.urdf, self.sim_timestep, q_init)
        self.WBID.setupProblem()

        # Setup the LIPM MPC
        ns = 20
        T = 1.
        self.lip_mpc = LIPMpc.LipController(q_init, ns, T)
        self.step_counter = 0
        self.sol_dict = None
        self.lip_state = None

        self.initial_com_pos = self.WBID.com.getActualPose()

    def permute_muj_to_xbi(self, xbi_qpos):
        """Convert mujoco qpos to xbi qpos."""
        # Place the 4th element at the 7th position
        xbi_qpos_perm = xbi_qpos.copy()
        xbi_qpos_perm[6] = xbi_qpos[3]
        xbi_qpos_perm[3] = xbi_qpos[6]
        return xbi_qpos_perm

    def global_to_local(self, xbi_qvel, xbi_qpos):
        """Convert global qvel to local qvel. USE MUJOCO QPOS"""
        qvel = xbi_qvel.copy()
        qpos = xbi_qpos.copy()
        quat = np.zeros(4)
        quat[0] = qpos[6]
        quat[1] = qpos[3]
        quat[2] = qpos[4]
        quat[3] = qpos[5]
        rotation = R.from_quat(quat)
        w_R_b = rotation.as_matrix()
        qvel_local = w_R_b.T @ qvel[0:3]
        qvel[0:3] = qvel_local
        return qvel
        
        

       
    def sim_step(self):
        """Perform a single simulation step."""
        tic()

        if self.step_counter % 5 == 0:
            self.sol_dict = self.lip_mpc.get_solution(state=None, visualize=True)
            self.lip_state = self.sol_dict["state0"]

        # Integrate the mpc solution
        node = 0
        param_values_at_node = list()
        for i_params in self.lip_mpc.lip.prb.getParameters().values():
            for i_param in i_params.getValues():
                param_values_at_node.append(i_param[node])
        p = cs.vcat(param_values_at_node)

        self.lip_state = np.array(cs.DM(self.lip_mpc.simulation_euler_integrator(self.lip_state, self.sol_dict["input"], p)))

        # Send CoM reference to WBID
        A = 0.03
        fr = 1. * 2 * np.pi
        com_ref = self.initial_com_pos + np.array([0.0, A * np.sin(fr * self.sim_time), 0.0])
        com_vel_ref = np.array([0.0,            fr    * A * np.cos(fr * self.sim_time), 0.0])
        com_acc_ref = np.array([0.0,          - fr**2 * A * np.sin(fr * self.sim_time), 0.0])
        self.WBID.com.setReference(com_ref, com_vel_ref, com_acc_ref)
        
        #self.WBID.com.setReference(self.lip_state[0:3], self.lip_state[15:18], self.sol_dict["rddot0"])
        
        """# Set foot references
        i = 0
        for contact_task in self.WBID.contact_tasks:
            contact_task_reference_pose, contact_task_reference_vel, contact_task_reference_acc = contact_task.getReference()
            contact_task_reference_pose.translation = 0.5 * \
                (self.lip_state[3 + 6*i : 6 + 6*i] + \
                 self.lip_state[6 + 6*i : 9 + 6*i])
            contact_task_reference_vel[0:3] = (0.5 * (self.lip_state[18 + 6*i : 21 + 6*i] + self.lip_state[21 + 6*i : 24 + 6*i])).flatten()
            contact_task_reference_acc[0:3] = (0.5 * (self.sol_dict["input"][3 + 6*i : 6 + 6*i] + self.sol_dict["input"][6 + 6*i : 9 + 6*i])).flatten()
            
            contact_task.setReference(contact_task_reference_pose, contact_task_reference_vel, contact_task_reference_acc)
            if self.sol_dict["cdot_switch"][i * 2].getValues()[0][0] == 1:
                self.WBID.wrench_limits[2*i].releaseContact(False)
                self.WBID.wrench_limits[2*i+1].releaseContact(False)
            else:
                self.WBID.wrench_limits[2*i].releaseContact(True)
                self.WBID.wrench_limits[2*i+1].releaseContact(True)

            i = i + 1 """

        # Copmute inverse dynamics
        ddq, forces = self.WBID.stepProblem(self.permute_muj_to_xbi(self.data.qpos), self.data.qvel, self.sim_time)
        tau = self.WBID.getInverseDynamics()

        # Exclude floating base
        self.data.ctrl = tau[6:]

        # self.pass_count += 1
        # if self.pass_count >= 1:
        # exit()

       # print("toc() ===========================", toc())
        mujoco.mj_step(self.model, self.data)

        self.step_counter += 1

    def run(self):
        """Run simple real-time simulation."""
        prev_time = time.time()
        self.sim_time = 0.0
        self.pass_count = 0

        while self.running and not rospy.is_shutdown() and self.viewer.is_alive:
            # Get real time elapsed since last step
            #current_time = time.time()
            #elapsed_wall_time = current_time - prev_time
            #prev_time = current_time

            # Compute simulation time to advance (applying real-time factor)
            #sim_time_to_advance = elapsed_wall_time * self.real_time_factor

            # Calculate steps needed
            #steps = max(1, int(sim_time_to_advance / self.sim_timestep))

            # Step the simulation
            #for _ in range(steps):

            self.sim_step()
            self.sim_time += self.sim_timestep

            # Render the current state
            self.viewer.render()

if __name__ == "__main__":
    # Setup the simulation
    sim = G1MujocoSimulation(q_init)
    
    # Run the simulation     
    sim.run()

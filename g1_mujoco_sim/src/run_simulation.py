#!/usr/bin/env python3


import sys
import os
# To be able to import srbd_mpc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import time
import mujoco
import mujoco_viewer
import rospkg
import rospy
import numpy as np
from ttictoc import tic, toc
import tf

from g1_locomotion.g1_mujoco_sim.config.config import q_init
from wbid import WholeBodyID
from srbd_mpc import mpc

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

        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Set initial joint configuration
        self.data.qpos = self.permute_muj_to_xbi(q_init)

        # Real-time settings
        self.sim_timestep = self.model.opt.timestep
        self.real_time_factor = 0.5  # 1.0 = real time

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
       
    def sim_step(self):
        """Perform a single simulation step."""
        
        tic()
        # Set the initial state for the MPC
        q_euler = np.array(tf.transformations.euler_from_quaternion(self.permute_muj_to_xbi(self.data.qpos)[3:7]))
        com_vel = WBID.model.getCOMJacobian() @ self.data.qvel
        
        MPC.x0[0:3] = q_euler.copy().reshape(3, 1)  # Get the initial position of the base
        MPC.x0[3:6] = WBID.model.getCOM().reshape(3, 1)   # Get the initial position of the com
        MPC.x0[6:9] = self.data.qvel[3:6].reshape(3, 1)   # Get the initial rotation of the base
        MPC.x0[9:12] = com_vel.reshape(3, 1)   # Get the initial velocity of the com
        MPC.x0[12] = MPC.g
        
        # Setup reference horizon.  
        MPC.x_ref_hor[0, :] = MPC.x0[:].copy().reshape(13)
        MPC.x_ref_hor[0:, 3:6] = [5.26790425e-02, 7.44339342e-05, 5.97983255e-01]   
        MPC.x_ref_hor[0:, -1] = MPC.g
        
        print("x0", MPC.x0) 
        # print("x_ref_hor", MPC.x_ref_hor)

        # Current foot heel and toe positions
        left_heel = WBID.model.getPose("left_foot_line_contact_lower").translation
        left_toe = WBID.model.getPose("left_foot_line_contact_upper").translation
        
        right_heel = WBID.model.getPose("right_foot_line_contact_lower").translation
        right_toe = WBID.model.getPose("right_foot_line_contact_upper").translation

        c_horizon = []
        contact_horizon = []
        for i in range(MPC.HORIZON_LENGTH):
            c_horizon.append(np.concatenate((left_heel, left_toe, right_heel ,right_toe)))
        
        # Both feet in contact for all the horizon
        for i in range(MPC.HORIZON_LENGTH):
            contact_horizon.append(np.array([1, 1, 1, 1]))

        p_com_horizon = MPC.x_ref_hor[:, 3:6].copy()

        # Solve the MPC
        u_opt0, x_opt1 = MPC.update(contact_horizon, c_horizon, p_com_horizon, x_current = MPC.x0 , one_rollout = True)
        print("toc() ===========================", toc())

        WBID.updateModel(self.permute_muj_to_xbi(self.data.qpos), self.data.qvel)
        WBID.stack.update()
        WBID.setReference(x_opt1[1, 3:6].flatten(), u_opt0.flatten(), self.sim_time)
        WBID.solveQP()

        # WBID.stepProblem(self.permute_muj_to_xbi(self.data.qpos), self.data.qvel, self.sim_time)

        
        tau = WBID.getInverseDynamics()

        # Exclude floating base
        self.data.ctrl = tau[6:]

        self.pass_count += 1
        if self.pass_count >= 2000:
            exit()

        mujoco.mj_step(self.model, self.data)

    def run(self):
        """Run simple real-time simulation."""
        prev_time = time.time()
        self.sim_time = 0.0
        self.pass_count = 0
        
        # try:
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

        # except Exception as e:
        #     rospy.logerr(f"Simulation error: {e}")
        # finally:
        #     if self.viewer:
        #         self.viewer.close()


if __name__ == "__main__":
    
    # Setup the simulation
    sim = G1MujocoSimulation(q_init)
    

    # Setup the MPC
    MPC = mpc.MPC(dt = 0.04)
    MPC.init_matrices()

    # Setup the whole body ID
    WBID = WholeBodyID(sim.urdf, sim.sim_timestep, q_init)
    WBID.setupProblem()
    
    # Run the simulation     
    sim.run()

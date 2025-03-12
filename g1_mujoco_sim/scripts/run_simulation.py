#!/usr/bin/env python3

import os
import time
import mujoco
import mujoco_viewer
import rospkg
import rospy
import numpy as np
from ttictoc import tic, toc

from config import q_init
from wbid import WholeBodyID

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
        self.real_time_factor = 1.0  # 1.0 = real time

        # Create viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Running flag
        self.running = True

        # Setup the whole body ID
        self.WBID = WholeBodyID(self.urdf, self.sim_timestep, q_init)
        self.WBID.setupProblem()

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
        ddq, forces = self.WBID.stepProblem(self.permute_muj_to_xbi(self.data.qpos), self.data.qvel, self.sim_time)

        tau = self.WBID.getInverseDynamics()

        # Exclude floating base
        self.data.ctrl = tau[6:]

        # self.pass_count += 1
        # if self.pass_count >= 1:
        # exit()

        print("toc() ===========================", toc())
        mujoco.mj_step(self.model, self.data)

    def run(self):
        """Run simple real-time simulation."""
        prev_time = time.time()
        self.sim_time = 0.0
        self.pass_count = 0

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

if __name__ == "__main__":
    # Setup the simulation
    sim = G1MujocoSimulation(q_init)
    
    # Run the simulation     
    sim.run()

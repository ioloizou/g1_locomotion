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
from std_msgs.msg import String, Header
from g1_msgs.msg import SRBD_state
from geometry_msgs.msg import Vector3

from config import q_init
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
       
    def sim_step(self):
        """Perform a single simulation step."""
        
        tic()
        # print("x_ref_hor", MPC.x_ref_hor)

        # Current foot heel and toe positions
        left_heel = WBID.model.getPose("left_foot_line_contact_lower").translation
        left_toe = WBID.model.getPose("left_foot_line_contact_upper").translation
        
        right_heel = WBID.model.getPose("right_foot_line_contact_lower").translation
        right_toe = WBID.model.getPose("right_foot_line_contact_upper").translation

        WBID.updateModel(self.permute_muj_to_xbi(self.data.qpos), self.data.qvel)
        WBID.stack.update()
        WBID.setReference(self.sim_time)
        WBID.solveQP()

        
        tau = WBID.getInverseDynamics()
        print("toc() ===========================", toc())

        # Exclude floating base
        self.data.ctrl = tau[6:]

        # self.pass_count += 1
        # if self.pass_count >= 2000:
        #     exit()

        mujoco.mj_step(self.model, self.data)

    def run(self):
        """Run simple real-time simulation."""
        prev_time = time.time()
        self.sim_time = 0.0
        self.pass_count = 0
        
        pub = rospy.Publisher("hello_pub_test", String, queue_size=10)
        
        pub_srbd = rospy.Publisher("srbd_pub_test", SRBD_state, queue_size=10)
        
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
                hello_str = "Hello from ROS! %s" % rospy.get_time()
                pub.publish(hello_str)
            
                srbd_state_msg = SRBD_state()

                srbd_state_msg.header.stamp = rospy.Time.now()
                
                srbd_state_msg.header.frame_id = "g1_mujoco_sim"
                
                srbd_state_msg.header.frame_id = "hello_frame"
                srbd_state_msg.position.x = 2
                srbd_state_msg.position.y = 2
                srbd_state_msg.position.z = 2
                srbd_state_msg.orientation.x = 3
                srbd_state_msg.orientation.y = 3
                srbd_state_msg.orientation.z = 3

                pub_srbd.publish(srbd_state_msg)

            # rate.sleep()                
            # Render the current state
            self.viewer.render()

        

if __name__ == "__main__":
    try:
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
    
    except rospy.ROSInterruptException:
        pass
    # except KeyboardInterrupt:
    #     print("Simulation interrupted by user.")
    # finally:
    #     if sim.viewer:
    #         sim.viewer.close()
    #     rospy.signal_shutdown("Simulation closed.")
    #     # rospy.loginfo("Simulation closed.")
    #     # rospy.signal_shutdown("Simulation closed.")
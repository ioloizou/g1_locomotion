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
import tf
import numpy as np
from ttictoc import tic, toc

from std_msgs.msg import String, Header
from geometry_msgs.msg import Vector3
from g1_msgs.msg import SRBD_state
from g1_msgs.msg import ContactPoint

from config import q_init
from wbid import WholeBodyID
from srbd_mpc import mpc

def publish_current_state(pub_srbd, 
                          srbd_state_msg, 
                          base_orientation_curr, 
                          com_position_curr, 
                          base_angular_velocity_curr, 
                          com_linear_velocity_curr,
                          foot_positions_curr 
                          ):
    """
    Publish the current state of the robot to the ROS topic.
    """
    
    srbd_state_msg.header.stamp = rospy.Time.now()
    
    srbd_state_msg.orientation.x = base_orientation_curr[0]
    srbd_state_msg.orientation.y = base_orientation_curr[1]
    srbd_state_msg.orientation.z = base_orientation_curr[2]
    
    srbd_state_msg.position.x = com_position_curr[0]
    srbd_state_msg.position.y = com_position_curr[1]
    srbd_state_msg.position.z = com_position_curr[2]
    
    srbd_state_msg.angular_velocity.x = base_angular_velocity_curr[0]
    srbd_state_msg.angular_velocity.y = base_angular_velocity_curr[1]
    srbd_state_msg.angular_velocity.z = base_angular_velocity_curr[2]
    
    srbd_state_msg.linear_velocity.x = com_linear_velocity_curr[0]
    srbd_state_msg.linear_velocity.y = com_linear_velocity_curr[1]
    srbd_state_msg.linear_velocity.z = com_linear_velocity_curr[2]

    for i, contact_name in enumerate(["left_foot_line_contact_lower", "left_foot_line_contact_upper", "right_foot_line_contact_lower", "right_foot_line_contact_upper"]):
        
        contact_point = ContactPoint()  
        
        contact_point.name = String(contact_name)
        contact_point.position.x = Vector3(foot_positions_curr[i*3:i*3+3])
        contact_point.position.y = Vector3(foot_positions_curr[i*3:i*3+3])
        contact_point.position.z = Vector3(foot_positions_curr[i*3:i*3+3])

        # Force and contact state are not needed to pub to MPC
        srbd_state_msg.contact_points.append(contact_point)

    pub_srbd.publish(srbd_state_msg)


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
       
    def sim_step(self, pub_srbd, srbd_state_msg):
        """Perform a single simulation step."""
        
        tic()
        # print("x_ref_hor", MPC.x_ref_hor)

        # Current foot heel and toe positions
        left_heel = WBID.model.getPose("left_foot_line_contact_lower").translation
        left_toe = WBID.model.getPose("left_foot_line_contact_upper").translation
        
        right_heel = WBID.model.getPose("right_foot_line_contact_lower").translation
        right_toe = WBID.model.getPose("right_foot_line_contact_upper").translation

        foot_positions_curr = np.array([left_heel, left_toe, right_heel, right_toe])

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

        # Publish the current state
        base_orientation_curr = tf.transformations.euler_from_quaternion(self.permute_muj_to_xbi(self.data.qpos[3:7]))
        com_position_curr = WBID.com.getReference()
        base_angular_velocity_curr = self.data.qvel[3:6]
        com_linear_velocity_curr =  WBID.model.getCOMJacobian() @ self.data.qvel
        publish_current_state(pub_srbd,
                              srbd_state_msg,
                              base_orientation_curr,
                              com_position_curr,
                              base_angular_velocity_curr,
                              com_linear_velocity_curr,
                              foot_positions_curr
                              )

    def run(self):
        """Run simple real-time simulation."""
        prev_time = time.time()
        self.sim_time = 0.0
        self.pass_count = 0
                
        pub_srbd = rospy.Publisher("srbd_pub_test", SRBD_state, queue_size=10)
        
        srbd_state_msg = SRBD_state()

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
                self.sim_step(pub_srbd, srbd_state_msg)
                self.sim_time += self.sim_timestep

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
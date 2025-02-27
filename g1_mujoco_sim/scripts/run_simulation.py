#!/usr/bin/env python3

import os
import time
import numpy as np
import mujoco
import mujoco_viewer
import rospkg
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped
import tf2_ros

class G1MujocoSimulation:
    def __init__(self):
        rospy.init_node('g1_mujoco_sim', anonymous=True)
        
        # Find the model path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('g1_mujoco_sim')
        # Get the path to the g1_description package
        model_path = os.path.join(pkg_path, '..', 'g1_description', 'g1_23dof.xml')
        
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Create the viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # Publishers
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Initialize messages
        self.joint_state_msg = JointState()
        self.joint_state_msg.name = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                                    for i in range(self.model.njnt) if self.model.jnt_type[i] != 0]  # Exclude free joints
        
        # Control rate
        self.rate = rospy.Rate(100)  # 100 Hz
        
    def run(self):
        """Run the simulation loop."""
        try:
            while not rospy.is_shutdown() and self.viewer.is_alive:
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update the viewer
                self.viewer.render()
                
                # Publish joint states
                self.publish_joint_states()
                
                # Publish base transform
                self.publish_base_transform()
                
                # Sleep to maintain the desired rate
                self.rate.sleep()
                
        except Exception as e:
            rospy.logerr(f"Simulation error: {e}")
        finally:
            self.viewer.close()
            
    def publish_joint_states(self):
        """Publish joint states to ROS."""

    def publish_base_transform(self):
        """Publish the base transform to TF."""
        # Find the body ID for the base
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_id >= 0:
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "world"
            transform.child_frame_id = "base_link"
            # Get position from simulation data
            pos = self.data.xpos[base_id]
            transform.transform.translation.x = pos[0]
            transform.transform.translation.y = pos[1]
            transform.transform.translation.z = pos[2]
            # Get orientation from simulation data (w, x, y, z)
            quat = self.data.xquat[base_id]
            transform.transform.rotation.w = quat[0]
            transform.transform.rotation.x = quat[1]
            transform.transform.rotation.y = quat[2]
            transform.transform.rotation.z = quat[3]
            self.tf_broadcaster.sendTransform(transform)

if __name__ == "__main__":
    sim = G1MujocoSimulation()
    sim.run()
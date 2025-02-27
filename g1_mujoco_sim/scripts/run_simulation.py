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
import threading

class G1MujocoSimulation:
    def __init__(self):
        rospy.init_node('g1_mujoco_sim', anonymous=True)
        
        # Find the model path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('g1_mujoco_sim')
        model_path = os.path.join(pkg_path, '..', 'g1_description', 'g1_23dof.xml')
        
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        
        # Initialize data
        self.data = mujoco.MjData(self.model)
        
        # Create viewer
        self.viewer = mujoco_viewer.MujocoViewer(
            self.model, 
            self.data,
        )
        
        # Publishers
        self.joint_pub = rospy.Publisher('joint_states', JointState, queue_size=1)  
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Initialize messages
        self.joint_state_msg = JointState()
        self.joint_state_msg.name = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                                    for i in range(self.model.njnt) if self.model.jnt_type[i] != 0]
        
        # Significant performance settings
        self.sim_rate = 400  # Hz for simulation
        self.render_every_n_steps = 4  # Render 1/4 of frames
        self.publish_every_n_steps = 2  # Publish 1/2 of frames
        self.step_counter = 0
        
        # Use separate thread for physics
        self.running = True
        self.physics_thread = True
        
        # Rate limiter
        self.rate = rospy.Rate(self.sim_rate)
        
    def run(self):
        """Run the simulation with multi-threading."""
        try:
            # Start physics in separate thread for better performance
            self.physics_thread = threading.Thread(target=self.physics_loop)
            self.physics_thread.daemon = True
            self.physics_thread.start()
            
            # Main thread handles rendering
            self.render_loop()
                
        except Exception as e:
            rospy.logerr(f"Simulation error: {e}")
        finally:
            self.running = False
            if self.physics_thread:
                self.physics_thread.join(timeout=1.0)
            self.viewer.close()
    
    def physics_loop(self):
        """Run physics simulation in separate thread."""
        last_time = time.time()
        fps_counter = 0
        
        while self.running and not rospy.is_shutdown():
            # Step physics
            mujoco.mj_step(self.model, self.data)
            self.step_counter += 1
            
            # Publish only every N steps
            if self.step_counter % self.publish_every_n_steps == 0:
                self.publish_joint_states()
                self.publish_base_transform()
            
            # FPS calculation
            fps_counter += 1
            if time.time() - last_time > 5:
                fps = fps_counter / (time.time() - last_time)
                rospy.loginfo(f"Physics simulation rate: {fps:.1f} Hz")
                fps_counter = 0
                last_time = time.time()
                
            # Control rate
            self.rate.sleep()
    
    def render_loop(self):
        """Handle rendering separately."""
        render_rate = rospy.Rate(self.sim_rate / self.render_every_n_steps)
        
        while self.running and not rospy.is_shutdown() and self.viewer.is_alive:
            # Update the viewer less frequently than physics
            self.viewer.render()
            render_rate.sleep()
            
    def publish_joint_states(self):
        """Publish joint states to ROS."""
        self.joint_state_msg.header.stamp = rospy.Time.now()
        
        # Get joint positions and velocities (optimized)
        joint_positions = []
        joint_velocities = []
        
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != 0:  # Exclude free joints
                joint_positions.append(self.data.qpos[i])
                joint_velocities.append(self.data.qvel[i])
        
        self.joint_state_msg.position = joint_positions
        self.joint_state_msg.velocity = joint_velocities
        
        # Publish
        self.joint_pub.publish(self.joint_state_msg)

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
            
            # Get orientation from simulation data
            quat = self.data.xquat[base_id]
            transform.transform.rotation.w = quat[0]
            transform.transform.rotation.x = quat[1]
            transform.transform.rotation.y = quat[2]
            transform.transform.rotation.z = quat[3]
            
            self.tf_broadcaster.sendTransform(transform)

if __name__ == "__main__":
    # Set threading model
    os.environ["OMP_NUM_THREADS"] = "4"  # Optimize OpenMP threading
    
    sim = G1MujocoSimulation()
    sim.run()
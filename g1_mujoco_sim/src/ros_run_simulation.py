#!/usr/bin/env python3
import os
import time
import mujoco
import mujoco_viewer
import rospkg
import rospy
import tf
import numpy as np
from ttictoc import tic, toc

from g1_msgs.msg import SRBD_state, ContactPoint

from config import q_init
from wbid import WholeBodyID

def publish_current_state(pub_srbd, 
                          srbd_state_msg,
                          contact_point_msg, 
                          base_orientation_curr, 
                          com_position_curr, 
                          base_angular_velocity_curr, 
                          com_linear_velocity_curr,
                          foot_positions_curr 
                          ):
    """
    Publish the current state of the robot to the ROS topic.
    """
    
    srbd_state_msg.contacts = []
    srbd_state_msg.header.stamp = rospy.Time.now()
    srbd_state_msg.header.frame_id = "SRBD"
    
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

    # I need to load gravity from mujoco and also in MPC
    srbd_state_msg.gravity = -9.80665
    
    for i, contact_name in enumerate(["left_foot_line_contact_lower", "left_foot_line_contact_upper", "right_foot_line_contact_lower", "right_foot_line_contact_upper"]):
        contact_point_msg = ContactPoint()
        contact_point_msg.name = contact_name
        contact_point_msg.position.x = foot_positions_curr[i, 0]
        contact_point_msg.position.y = foot_positions_curr[i, 1]
        contact_point_msg.position.z = foot_positions_curr[i, 2]

        srbd_state_msg.contacts.append(contact_point_msg)
    
    
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

        self.x_opt1 = np.zeros(13)  
        self.u_opt0 = np.zeros(12)    

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
        
        # Set mujoco timestep
        self.model.opt.timestep = 0.003  # Set the simulation timestep
        # Real-time settings
        self.sim_timestep = self.model.opt.timestep

        self.real_time_factor = 0.5  # 1.0 = real time

        # Create viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Running flag
        self.running = True

    def callback_mpc_solution(self, msg):
        """
        Subscribe to the MPC solution and update the reference trajectory.
        """
        # Unpack the MPC solution from the incoming message
        
        # Everything is in world frame
        self.x_opt1[0] = msg.orientation.x
        self.x_opt1[1] = msg.orientation.y
        self.x_opt1[2] = msg.orientation.z
        self.x_opt1[3] = msg.position.x
        self.x_opt1[4] = msg.position.y
        self.x_opt1[5] = msg.position.z
        self.x_opt1[6] = msg.angular_velocity.x
        self.x_opt1[7] = msg.angular_velocity.y
        self.x_opt1[8] = msg.angular_velocity.z
        self.x_opt1[9] = msg.linear_velocity.x
        self.x_opt1[10] = msg.linear_velocity.y
        self.x_opt1[11] = msg.linear_velocity.z
        self.x_opt1[12] = msg.gravity

        # Get the optimal GRF from contact point
        for i, contact_point in enumerate(msg.contacts):
            self.u_opt0[i * 3: i * 3 + 3] = [contact_point.force.x, contact_point.force.y, contact_point.force.z]

        print("u_opt0\n", self.u_opt0)

    def permute_muj_to_xbi(self, xbi_qpos):
        """Convert mujoco qpos to xbi qpos."""
        # Place the 4th element at the 7th position
        xbi_qpos_perm = xbi_qpos.copy()
        xbi_qpos_perm[6] = xbi_qpos[3]
        xbi_qpos_perm[3] = xbi_qpos[6]
        return xbi_qpos_perm
       
    def sim_step(self, pub_srbd, srbd_state_msg, contact_point_msg):
        """Perform a single simulation step."""
        
        tic()

        # Current foot heel and toe positions in world frame
        left_heel = WBID.model.getPose("left_foot_line_contact_lower").translation
        left_toe = WBID.model.getPose("left_foot_line_contact_upper").translation
        right_heel = WBID.model.getPose("right_foot_line_contact_lower").translation
        right_toe = WBID.model.getPose("right_foot_line_contact_upper").translation

        foot_positions_curr = np.array([left_heel, left_toe, right_heel, right_toe])

        # Floating base: 
        # Xbot/Pinocchio - Position: World frame            Mujoco - Position: World frame
        # Xbot/Pinocchio - Orientation: World frame         Mujoco - Orientation: World frame    
        # Xbot/Pinocchio - Linear Velocity: Local frame     Mujoco - Linear Velocity: World frame
        # Xbot/Pinocchio - Angular Velocity: Local frame    Mujoco - Angular Velocity: Local frame

        # Transform the mujoco linear velocity to the xbot linear velocity
        # World to Local | Note: linear is the rotation matrix
        
        permuted_qpos = self.permute_muj_to_xbi(self.data.qpos)
        quat = permuted_qpos[3:7]
        w_Rot_b = tf.transformations.quaternion_matrix(quat)[0:3, 0:3]
        base_linear_velocity_local = w_Rot_b.T @ self.data.qvel[0:3]
        joints_velocity_local = np.concatenate([base_linear_velocity_local, self.data.qvel[3:]])

        WBID.updateModel(permuted_qpos, joints_velocity_local)
        WBID.stack.update()
        WBID.setReference(self.sim_time, self.x_opt1, self.u_opt0)
        # WBID.setReference(self.sim_time)
        WBID.solveQP()

        
        tau = WBID.getInverseDynamics()
        print("toc() ===========================", toc())

        # Exclude floating base
        self.data.ctrl = tau[6:]

        # self.pass_count += 1
        # if self.pass_count >= 10000:
        #     exit()

        mujoco.mj_step(self.model, self.data)
        
        permuted_qpos = self.permute_muj_to_xbi(self.data.qpos)
        quat = permuted_qpos[3:7]
        w_Rot_b = tf.transformations.quaternion_matrix(quat)[0:3, 0:3]
        base_linear_velocity_local = w_Rot_b.T @ self.data.qvel[0:3]
        joints_velocity_local = np.concatenate([base_linear_velocity_local, self.data.qvel[3:]])
        WBID.updateModel(permuted_qpos, joints_velocity_local)
        
        #### Maybe another way to get the CoM
        # com_position_curr = data.comPos
        # com_linear_velocity_curr = data.comVel 
        
        # Publish the current state
        permuted_qpos = self.permute_muj_to_xbi(self.data.qpos)
        # Maybe I need the torso orientation not the floating base
        
        # World Frame
        base_orientation_curr = tf.transformations.euler_from_matrix(w_Rot_b @ WBID.model.getPose("torso_link").linear)
        # World Frame
        com_position_curr = WBID.model.getCOM()
        # Local Frame -> World Frame
        base_angular_velocity_curr = w_Rot_b @ WBID.model.getVelocityTwist("torso_link")[3:6]
        # Local Frame -> World Frame
        com_linear_velocity_curr =  w_Rot_b @ WBID.model.getCOMJacobian() @ joints_velocity_local
        
        publish_current_state(pub_srbd,
                              srbd_state_msg,
                              contact_point_msg,
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

        # Create a message for the SRBD state        
        srbd_state_msg = SRBD_state()
        contact_point_msg = ContactPoint()

        # Create a publisher for the SRBD state        
        pub_srbd = rospy.Publisher("/srbd_current", SRBD_state, queue_size=10)

        # Create a subscriber for the MPC solution - fixed by using the class method
        sub_mpc = rospy.Subscriber("/mpc_solution", SRBD_state, self.callback_mpc_solution)

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
                self.sim_step(pub_srbd, srbd_state_msg, contact_point_msg)
                self.sim_time += self.sim_timestep

            # Render the current state
            self.viewer.render()

        

if __name__ == "__main__":
    try:
        # Setup the simulation
        sim = G1MujocoSimulation(q_init)

        # Setup the whole body ID
        WBID = WholeBodyID(sim.urdf, sim.sim_timestep, q_init)
        WBID.setupProblem()
        # Run the simulation     
        sim.run()
    
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        if sim.viewer:
            sim.viewer.close()
        rospy.signal_shutdown("Simulation closed.")
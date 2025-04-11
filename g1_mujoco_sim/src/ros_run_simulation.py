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
from pal_statistics import StatisticsRegistry
from rosgraph_msgs.msg import Clock
from g1_msgs.msg import SRBD_state, ContactPoint, Feet_reference, State

from config import q_init
from wbid import WholeBodyID
import viz


def publish_current_state(pub_srbd, 
                          contact_point_msg, 
                          base_orientation_curr, 
                          com_position_curr, 
                          base_angular_velocity_curr, 
                          com_linear_velocity_curr,
                          foot_positions_curr,
                          qp_forces,
                          sim_time=0.0
                          ):
    """
    Publish the current state of the robot to the ROS topic.
    """
    srbd_state_msg = SRBD_state()    
    srbd_state_msg.contacts = []
    srbd_state_msg.header.stamp = rospy.Time.from_sec(sim_time)
    srbd_state_msg.header.frame_id = "SRBD"
    
    state_msg = State()
    state_msg.trajectory_index = 0
    state_msg.orientation.x = base_orientation_curr[0]
    state_msg.orientation.y = base_orientation_curr[1]
    state_msg.orientation.z = base_orientation_curr[2]
    
    state_msg.position.x = com_position_curr[0]
    state_msg.position.y = com_position_curr[1]
    state_msg.position.z = com_position_curr[2]
    
    state_msg.angular_velocity.x = base_angular_velocity_curr[0]
    state_msg.angular_velocity.y = base_angular_velocity_curr[1]
    state_msg.angular_velocity.z = base_angular_velocity_curr[2]
    
    state_msg.linear_velocity.x = com_linear_velocity_curr[0]
    state_msg.linear_velocity.y = com_linear_velocity_curr[1]
    state_msg.linear_velocity.z = com_linear_velocity_curr[2]

    # I need to load gravity from mujoco and also in MPC
    state_msg.gravity = -9.80665
    
    srbd_state_msg.states_horizon.append(state_msg)
    
    # The same order as in the contact frames
    # foot_offset_x = [-0.05, 0.12, -0.05, 0.12]

    for i, contact_name in enumerate(["left_foot_line_contact_lower", "left_foot_line_contact_upper", "right_foot_line_contact_lower", "right_foot_line_contact_upper"]):
        contact_point_msg = ContactPoint()
        contact_point_msg.name = contact_name
        contact_point_msg.position.x = foot_positions_curr[i, 0]
        contact_point_msg.position.y = foot_positions_curr[i, 1]
        contact_point_msg.position.z = foot_positions_curr[i, 2]
        qp_forces = np.array(qp_forces).reshape(-1).copy()
        contact_point_msg.force.x = qp_forces[i * 3]
        contact_point_msg.force.y = qp_forces[i * 3 + 1]
        contact_point_msg.force.z = qp_forces[i * 3 + 2]

        srbd_state_msg.contacts.append(contact_point_msg)
    
    
    pub_srbd.publish(srbd_state_msg)

## Wrong need to recheck offsets
# def publish_feet_reference(pub_reference_feet_position,
#                            feet_ref_pos_msg,
#                            foot,
#                            swing_task_reference_pose,
#                            foot_positions_curr,
#                            sim_time):
#     # Reference pose for the swing foot to publish
#     feet_ref_pos_list = []
#     if foot == "left":    
#         foot_ref_pos = ContactPoint()       
#         foot_ref_pos.name = "right_foot_point_contact" 
#         foot_ref_pos.position.x = swing_task_reference_pose.translation[0]
#         foot_ref_pos.position.y = swing_task_reference_pose.translation[1]
#         foot_ref_pos.position.z = swing_task_reference_pose.translation[2]
        
#         feet_ref_pos_list.append(foot_ref_pos)

#         foot_ref_pos.name = "left_foot_point_contact"
#         foot_ref_pos.position.x = foot_positions_curr[0, 0]
#         foot_ref_pos.position.y = foot_positions_curr[0, 1]
#         foot_ref_pos.position.z = foot_positions_curr[0, 2]
#         feet_ref_pos_list.append(foot_ref_pos)
#     elif foot == "right":
#         foot_ref_pos = ContactPoint()
#         foot_ref_pos.name = "left_foot_point_contact"
#         foot_ref_pos.position.x = swing_task_reference_pose.translation[0]
#         foot_ref_pos.position.y = swing_task_reference_pose.translation[1]
#         foot_ref_pos.position.z = swing_task_reference_pose.translation[2]

#         feet_ref_pos_list.append(foot_ref_pos)
#         foot_ref_pos.name = "right_foot_point_contact"
#         foot_ref_pos.position.x = foot_positions_curr[1, 0]
#         foot_ref_pos.position.y = foot_positions_curr[1, 1]
#         foot_ref_pos.position.z = foot_positions_curr[1, 2]

    
#     feet_ref_pos_msg.header.stamp = rospy.Time.from_sec(sim_time)
#     feet_ref_pos_msg.header.frame_id = "world"
#     feet_ref_pos_msg.feet_positions = feet_ref_pos_list
#     pub_reference_feet_position.publish(feet_ref_pos_msg)

def loadURDF(pkg_path):
    with open(
        os.path.join(pkg_path, "..", "g1_description", "g1_23dof.urdf"), "r"
    ) as urdf_file:
        urdf = urdf_file.read()
    return urdf

class G1MujocoSimulation:
    def __init__(self, q_init):
        rospy.init_node("g1_mujoco_sim", anonymous=True)
        
        # x_opt is temporary
        self.x_opt = np.zeros((2, 13))
        self.u_opt0 = np.zeros(12)
        self.contact_states = np.zeros(4)    

        self.srbd_recieved = SRBD_state()
        
        # Add swing state tracking variables
        self.current_swing_foot = None
        self.is_swing_time_set = False
        self.start_swing_time = 0.0
        self.end_swing_time = 0.0
        self.last_swing_time = 0.0
        self.swing_duration = 0.25 

        # Find the model path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("g1_mujoco_sim")
        model_path = os.path.join(pkg_path, "..", "g1_description", "g1_23dof.xml")
        
        # Load the URDF for xbot
        self.urdf = loadURDF(pkg_path)

        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Set initial joint configuration. Xbot quaternion is [x, y, z, w] to mujoco [w, x, y, z]
        self.data.qpos = q_init.copy()
        self.data.qpos[3] = q_init[6]
        self.data.qpos[4] = q_init[3]
        self.data.qpos[5] = q_init[4]
        self.data.qpos[6] = q_init[5]
        
        # Set mujoco timestep
        self.model.opt.timestep = 0.001  # Set the simulation timestep
        # Real-time settings
        self.sim_timestep = self.model.opt.timestep

        # Create viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Setup rviz
        self.rviz_srbd_full = viz.RvizSrbdFullBody(self.urdf, ["left_foot_line_contact_lower", "left_foot_line_contact_upper", "right_foot_line_contact_lower", "right_foot_line_contact_upper"])

        # Add simulation time publisher
        self.pub_sim_time = rospy.Publisher("/simulation_time", Clock, queue_size=10)

        # Running flag
        self.running = True

    def callback_mpc_solution(self, msg):
        """
        Subscribe to the MPC solution and update the reference trajectory.
        """
        self.srbd_recieved = msg

        # Unpack the MPC solution from the incoming message
        self.x_opt = np.zeros((len(msg.states_horizon), 13))
        
        # Everything is in world frame
        for i in range(len(msg.states_horizon)):
            self.x_opt[i, 0] = msg.states_horizon[i].orientation.x
            self.x_opt[i, 1] = msg.states_horizon[i].orientation.y
            self.x_opt[i, 2] = msg.states_horizon[i].orientation.z
            self.x_opt[i, 3] = msg.states_horizon[i].position.x
            self.x_opt[i, 4] = msg.states_horizon[i].position.y
            self.x_opt[i, 5] = msg.states_horizon[i].position.z
            self.x_opt[i, 6] = msg.states_horizon[i].angular_velocity.x
            self.x_opt[i, 7] = msg.states_horizon[i].angular_velocity.y
            self.x_opt[i, 8] = msg.states_horizon[i].angular_velocity.z
            self.x_opt[i, 9] = msg.states_horizon[i].linear_velocity.x
            self.x_opt[i, 10] = msg.states_horizon[i].linear_velocity.y
            self.x_opt[i, 11] = msg.states_horizon[i].linear_velocity.z
            self.x_opt[i, 12] = msg.states_horizon[i].gravity

        # Get the optimal GRF from contact point
        for i, contact_point in enumerate(msg.contacts):
            self.u_opt0[i * 3: i * 3 + 3] = [contact_point.force.x, contact_point.force.y, contact_point.force.z]
            self.contact_states[i] = contact_point.active
            
        # print("u_opt0\n", self.u_opt0)

    def permute_muj_to_xbi(self, xbi_qpos):
        """Convert mujoco qpos to xbi qpos."""
        # Place the 4th element at the 7th position
        xbi_qpos_perm = xbi_qpos.copy()
        xbi_qpos_perm[3] = xbi_qpos[4]
        xbi_qpos_perm[4] = xbi_qpos[5]
        xbi_qpos_perm[5] = xbi_qpos[6]
        xbi_qpos_perm[6] = xbi_qpos[3]
        return xbi_qpos_perm
       
    def sim_step(self, pub_srbd, pub_reference_feet_position, feet_ref_pos_msg, contact_point_msg):
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
        
        ###################################
        #      Contact enable/disable     #
        ###################################
        
        # Map the four contact points to the two contact tasks
        # Left foot contact task is active if either left foot contact point is active
        left_foot_active = self.contact_states[0] == 1 or self.contact_states[1] == 1
        # Right foot contact task is active if either right foot contact point is active
        right_foot_active = self.contact_states[2] == 1 or self.contact_states[3] == 1


        def feet_gait_procedure(foot):
            """Deploy the feet gait procedure based on active leg"""
            
            # Only set new swing times if we're not already in a swing or if we're switching feet
            if not self.is_swing_time_set or self.current_swing_foot != foot:
                self.start_swing_time = self.sim_time
                self.end_swing_time = self.start_swing_time + self.swing_duration
                self.last_swing_time = self.start_swing_time
                self.is_swing_time_set = True
                self.current_swing_foot = foot
                rospy.loginfo(f'Starting new swing for {foot} foot')
            
            # Check if the current swing is complete
            if self.is_swing_time_set and self.sim_time >= self.end_swing_time:
                # Reset the swing time and alternate to the other foot for the next cycle
                self.is_swing_time_set = False
                self.last_swing_time = self.sim_time
                # The other foot will be chosen on the next call
            
            # To match left, right with the order of the tasks
            if foot == "right":
                foot_in_swing, foot_in_contact = (0, 1)
                wrench_indexes_swing = [0, 1]
                wrench_indexes_contact = [2, 3] 
            else:
                foot_in_swing, foot_in_contact = (1, 0)
                wrench_indexes_swing = [2, 3]
                wrench_indexes_contact = [0, 1]
            
            # Contact related
            WBID.contact_tasks[foot_in_contact].setActive(True)
            WBID.swing_tasks[foot_in_contact].setActive(False)
            WBID.swing_tasks[foot_in_contact].reset()
            WBID.contact_tasks[foot_in_contact].reset()
            WBID.wrench_limits[wrench_indexes_contact[0]].setWrenchLimits(np.array([0.0, 0.0, 10.0]), np.array([1000.0, 1000.0, 1000.0]))
            WBID.wrench_limits[wrench_indexes_contact[1]].setWrenchLimits(np.array([0.0, 0.0, 10.0]), np.array([1000.0, 1000.0, 1000.0]))

            # Swing related                        
            WBID.contact_tasks[foot_in_swing].setActive(False)
            WBID.swing_tasks[foot_in_swing].setActive(True)
            # I need to reset because if i do setActive(True) it will not reset the reference
            WBID.contact_tasks[foot_in_swing].reset()
            WBID.swing_tasks[foot_in_swing].reset()
            WBID.wrench_limits[wrench_indexes_swing[0]].setWrenchLimits(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
            WBID.wrench_limits[wrench_indexes_swing[1]].setWrenchLimits(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
            swing_task_reference_pose, swing_task_reference_vel, swing_task_reference_acc = WBID.swing_tasks[foot_in_swing].getReference()
            
            # Maximum swing height
            max_swing_height = 0.01

            # Calculate swing progress (0.0 to 1.0)
            cycle_progress = (self.sim_time - self.start_swing_time) / self.swing_duration
            
            # Set the reference for the foot to rise or lower based on cycle progress
            if cycle_progress < 0.5:  # First half - swing up
                # Rise proportionately from 0 to max_swing_height
                delta_z = max_swing_height * (cycle_progress / 0.5)
                swing_task_reference_pose.translation += (0., 0., delta_z)
                WBID.swing_tasks[foot_in_swing].setReference(swing_task_reference_pose, swing_task_reference_vel, swing_task_reference_acc)
                rospy.loginfo(f'{foot} in contact, other foot swinging up, progress: {cycle_progress:.2f}')
            else:  # Second half - swing down
                # Fall proportionately from max_swing_height back to 0
                delta_z = max_swing_height * ((1.0 - cycle_progress) / 0.5)
                swing_task_reference_pose.translation += (0., 0., -delta_z)
                WBID.swing_tasks[foot_in_swing].setReference(swing_task_reference_pose, swing_task_reference_vel, swing_task_reference_acc)
                rospy.loginfo(f'{foot} in contact, other foot swinging down, progress: {cycle_progress:.2f}')
            
            # Publish the reference foot position
            # publish_feet_reference(pub_reference_feet_position,
            #                        feet_ref_pos_msg,
            #                        foot,
            #                        swing_task_reference_pose,
            #                        foot_positions_curr,
            #                        self.sim_time)

        if left_foot_active and right_foot_active:
            rospy.loginfo("Double support phase")
            WBID.swing_tasks[0].setActive(False)
            WBID.swing_tasks[1].setActive(False)
        elif left_foot_active:
            feet_gait_procedure("left")
        elif right_foot_active:
            feet_gait_procedure("right")

        # Log less frequently to avoid spamming the console
        if self.sim_time - getattr(self, 'last_log_time', 0.0) >= 0.5:
            rospy.loginfo(f'Current swing foot: {self.current_swing_foot}, Swing time set: {self.is_swing_time_set}')
            self.last_log_time = self.sim_time
        
        ###################################
        
        WBID.stack.update()
        WBID.setReference(self.sim_time, self.x_opt[1, :], self.u_opt0, foot_positions_curr)
        WBID.solveQP()

        
        tau = WBID.getInverseDynamics()
        self.wbid_solve_time = toc()

        # Exclude floating base
        self.data.ctrl = tau[6:]

        # self.pass_count += 1
        # if self.pass_count >= 2000:
        #     exit()

        mujoco.mj_step(self.model, self.data)
        
        permuted_qpos = self.permute_muj_to_xbi(self.data.qpos)
        quat = permuted_qpos[3:7]
        w_Rot_b = tf.transformations.quaternion_matrix(quat)[0:3, 0:3]
        base_linear_velocity_local = w_Rot_b.T @ self.data.qvel[0:3]
        joints_velocity_local = np.concatenate([base_linear_velocity_local, self.data.qvel[3:]])
        WBID.updateModel(permuted_qpos, joints_velocity_local)
        
        # Publish the current state
        #permuted_qpos = self.permute_muj_to_xbi(self.data.qpos)
        # Maybe I need the torso orientation not the floating base
        
        # World Frame
        base_orientation_curr = tf.transformations.euler_from_matrix(WBID.model.getPose("pelvis").linear)
        # World Frame
        com_position_curr = WBID.model.getCOM()
        # World Frame
        base_angular_velocity_curr = WBID.model.getVelocityTwist("pelvis")[3:6]
        # World Frame
        com_linear_velocity_curr = WBID.model.getCOMJacobian() @ joints_velocity_local

        # Get the forces from QP to publish
        publish_current_state(pub_srbd,
                              contact_point_msg,
                              base_orientation_curr,
                              com_position_curr,
                              base_angular_velocity_curr,
                              com_linear_velocity_curr,
                              foot_positions_curr,
                              WBID.contact_forces,
                              self.sim_time
                              )
        
        # Publish the joint state for rviz
        self.rviz_srbd_full.publishJointState(self.sim_time, WBID.model.getJointPosition())

        # Publish QP forces not SRBD forces for rviz
        for i, contact_frame in enumerate(WBID.contact_frames):
            self.rviz_srbd_full.publishContactForce(rospy.Time(self.sim_time), WBID.contact_forces[i], contact_frame)


        self.rviz_srbd_full.publishSRBDViewer(WBID.inertia_torso, self.srbd_recieved, rospy.Time(self.sim_time), WBID.contact_frames)

        # Publish the com horizon in rviz
        self.rviz_srbd_full.publishPointTrj(self.srbd_recieved.states_horizon, rospy.Time(self.sim_time), "CoM")

        # Publish the simulation time
        sim_time_msg = Clock()
        sim_time_msg.clock = rospy.Time.from_sec(self.sim_time)
        self.pub_sim_time.publish(sim_time_msg)

        # Publish the statistics
        self.registry.publish()


    def run(self):
        """Run simple real-time simulation."""
        self.sim_time = 0.0
        self.pass_count = 0

        # Create a message for the SRBD state        
        contact_point_msg = ContactPoint()
        feet_ref_pos_msg = Feet_reference()

        # Create a publisher for the SRBD state        
        pub_srbd = rospy.Publisher("/srbd_current", SRBD_state, queue_size=10)

        # Create a subscriber for the MPC solution - fixed by using the class method
        sub_mpc = rospy.Subscriber("/mpc_solution", SRBD_state, self.callback_mpc_solution)

        pub_reference_feet_position = rospy.Publisher("/feet_ref_pos", Feet_reference, queue_size=10)

        # Create Registry for a topic
        self.registry = StatisticsRegistry("/wbid_statistics")
        self.wbid_solve_time = 0.0
        self.registry.registerFunction("wbid_solve_time", (lambda: self.wbid_solve_time))

        while self.running and not rospy.is_shutdown() and self.viewer.is_alive:
            # Get real time elapsed since last step
            self.sim_step(pub_srbd, pub_reference_feet_position, feet_ref_pos_msg, contact_point_msg)
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
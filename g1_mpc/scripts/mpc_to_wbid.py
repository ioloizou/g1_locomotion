#!/usr/bin/env python3
from mpc import MPC
import numpy as np
from ttictoc import tic, toc

import rospy
from pal_statistics import StatisticsRegistry


from g1_msgs.msg import SRBD_state, ContactPoint
from std_msgs.msg import Header

def gait_planner(stance_duration=3.2):
	"""
	Very simple gait planner that alternates between two contact points every 0.4 seconds
	based on elapsed time.
	"""
	# Initialize static variables on first call
	if not hasattr(gait_planner, "last_switch_time"):
		gait_planner.last_switch_time = rospy.Time.now()
		gait_planner.current_phase = 0
	
	# Calculate elapsed time since last switch
	current_time = rospy.Time.now()
	elapsed = (current_time - gait_planner.last_switch_time).to_sec()
	
	# Check if it's time to switch gait pattern
	if elapsed >= stance_duration:
		gait_planner.current_phase = (gait_planner.current_phase + 1) % 2
		gait_planner.last_switch_time = current_time
	
	# Create contact horizon based on current phase
	contact_horizon = []
	for _ in range(MPC.HORIZON_LENGTH):
		if gait_planner.current_phase == 0:
			contact_horizon.append(np.array([1, 1, 0, 0])) 
		else:
			contact_horizon.append(np.array([0, 0, 1, 1]))
	
	return contact_horizon

def callback_srbd_current(msg):
	"""
	Callback function to update the current state of the robot.
	"""
	MPC.x0[0] = msg.orientation.x
	MPC.x0[1] = msg.orientation.y
	MPC.x0[2] = msg.orientation.z
	MPC.x0[3] = msg.position.x
	MPC.x0[4] = msg.position.y
	MPC.x0[5] = msg.position.z
	MPC.x0[6] = msg.angular_velocity.x
	MPC.x0[7] = msg.angular_velocity.y
	MPC.x0[8] = msg.angular_velocity.z
	MPC.x0[9] = msg.linear_velocity.x
	MPC.x0[10] = msg.linear_velocity.y
	MPC.x0[11] = msg.linear_velocity.z
	MPC.x0[12] = msg.gravity

	# Get current position of the contact points
	# Until footstep is implemented, we assume that the contact points are constant
	contact_positions = []
	for i in range(len(msg.contacts)):
		contact_positions.extend([msg.contacts[i].position.x,
								  msg.contacts[i].position.y,
								  msg.contacts[i].position.z])
	MPC.c_horizon[:] = np.tile(np.array(contact_positions), (MPC.HORIZON_LENGTH, 1))

def publish_mpc_solution():	
	"""Publish the MPC solution u_opt0 and x_opt1 to the ROS topic."""
	srbd_state_msg = SRBD_state()
	srbd_state_msg.contacts = []
	srbd_state_msg.header = Header()
	srbd_state_msg.header.stamp = rospy.Time.now()
	srbd_state_msg.header.frame_id = "SRBD"
	srbd_state_msg.orientation.x = MPC.x_opt[1, 0]
	srbd_state_msg.orientation.y = MPC.x_opt[1, 1]
	srbd_state_msg.orientation.z = MPC.x_opt[1, 2]
	srbd_state_msg.position.x = MPC.x_opt[1, 3]
	srbd_state_msg.position.y = MPC.x_opt[1, 4]
	srbd_state_msg.position.z = MPC.x_opt[1, 5]
	srbd_state_msg.angular_velocity.x = MPC.x_opt[1, 6]
	srbd_state_msg.angular_velocity.y = MPC.x_opt[1, 7]
	srbd_state_msg.angular_velocity.z = MPC.x_opt[1, 8]
	srbd_state_msg.linear_velocity.x = MPC.x_opt[1, 9]
	srbd_state_msg.linear_velocity.y = MPC.x_opt[1, 10]
	srbd_state_msg.linear_velocity.z = MPC.x_opt[1, 11]
	srbd_state_msg.gravity = MPC.x_opt[1, 12]

	for i, contact_name in enumerate(["left_foot_line_contact_lower", "left_foot_line_contact_upper", "right_foot_line_contact_lower", "right_foot_line_contact_upper"]):
		contact_point_msg = ContactPoint()
		contact_point_msg.name = contact_name
		contact_point_msg.force.x = MPC.u_opt[0, i * 3]
		contact_point_msg.force.y = MPC.u_opt[0, i * 3 + 1]
		contact_point_msg.force.z = MPC.u_opt[0, i * 3 + 2]
		
		# Contact State
		contact_point_msg.active = gait_planner()[0][i]
		srbd_state_msg.contacts.append(contact_point_msg)

	pub_mpc_solution.publish(srbd_state_msg)		

if __name__ == '__main__':
	rospy.init_node('mpc', anonymous=True)


	# Create Registry for a topic
	registry = StatisticsRegistry("/mpc_statistics")
	mpc_solve_time = 0.0
	registry.registerFunction("mpc_solve_time", (lambda: mpc_solve_time))

	MPC = MPC()
	MPC.init_matrices()
	
	sub_current_state = rospy.Subscriber('/srbd_current', SRBD_state, callback_srbd_current)
	pub_mpc_solution = rospy.Publisher('/mpc_solution', SRBD_state, queue_size=10)
	
	while not rospy.is_shutdown():	

		# Temporary until refactor of the code
		c_horizon = MPC.c_horizon.copy()
		
		contact_horizon = []
		# Both feet in contact for all the horizon
		for i in range(MPC.HORIZON_LENGTH):
			contact_horizon.append(np.array([1, 1, 1, 1]))

		# MPC.x_ref_hor[:, 3:6] = np.tile([5.26790425e-02, 7.44339342e-05, 5.97983255e-01], (MPC.HORIZON_LENGTH, 1))
		
		# Get current time in seconds
		time = rospy.Time.now().to_sec()
	
		# Update the reference trajectory:
		# Keep x constant, and update y and z to follow a time-varying circle

		# Define circle parameters that change by time

		MPC.x_ref_hor[0, 3:6] = [5.26790425e-02, 7.44339342e-05, 5.97983255e-01]

		radius = 0.05
		x_const = MPC.x_ref_hor[0, 3]
		y_center = MPC.x_ref_hor[0, 4]
		z_center = MPC.x_ref_hor[0, 5]

		speed = 0.5
		MPC.x_ref_hor[:, 3] = x_const 
		MPC.x_ref_hor[:, 4] = y_center 
		MPC.x_ref_hor[:, 5] = z_center 
		# - radius * np.cos(speed*np.pi*time)
		MPC.x_ref_hor[:, 12]= MPC.g

		p_com_horizon = MPC.x_ref_hor[:, 3:6].copy()
		
		# contact_horizon = gait_planner()

		# Update the MPC solution
		tic()
		MPC.update(contact_horizon, c_horizon, p_com_horizon, x_current=MPC.x0.copy(), one_rollout=True)
		mpc_solve_time = toc()
		
		registry.publish()

		# Publish the MPC solution
		publish_mpc_solution()

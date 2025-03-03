q_init = [
	# floating base
	0.0,
	0.0,
	0.793 - 0.117,  # reference base linear # Note i should do FW kinematics and subscrabt the difference of z from the default
	1.0, # reference base quaternion
	0.0,
	0.0,
	0.0,  
	## left leg
	-0.6,  # left_hip_pitch_joint
	0.0,  # left_hip_roll_joint
	0.0,  # left_hip_yaw_joint
	1.2,  # left_knee_joint
	-0.6,  # left_ankle_pitch_joint
	0.0,  # left_ankle_roll_joint
	## right leg
	-0.6,  # right_hip_pitch_joint
	0.0,  # right_hip_roll_joint
	0.0,  # right_hip_yaw_joint
	1.2,  # right_knee_joint
	-0.6,  # right_ankle_pitch_joint
	0.0,  # right_ankle_roll_joint
	## waist
	0.0,  # waist_yaw_joint
	## left shoulder
	0.0,  # left_shoulder_pitch_joint
	0.0,  # left_shoulder_roll_joint
	0.0,  # left_shoulder_yaw_joint
	0.0,  # left_elbow_joint
	0.0,  # left_wrist_roll_joint
	## right shoulder
	0.0,  #'right_shoulder_pitch_joint'
	0.0,  # right_shoulder_roll_joint
	0.0,  # right_shoulder_yaw_joint
	0.0,  # right_elbow_joint
	0.0,  # right_wrist_roll_joint
]

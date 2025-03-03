#!/usr/bin/env python3

import os
import time
import mujoco
import mujoco_viewer
import rospkg
import rospy
import pinocchio
import numpy as np

def PID_controller(data, q_desired):
    " A simple PID controller"

    # Ingoring floating base
    q_current = data.qpos[7:]
    q_desired = q_init[7:]
    dq_current = data.qvel[6:]
    dq_desired = np.zeros(dq_current.shape)

    # PID gains

    scale = 1.5 

    Kp = np.zeros(23)
    Kd = np.zeros(23)
    Kp[0:6] = [530.0, 570.0, 550.0, 270.0, 130.0, 30.0]
    Kd[0:6] = [60.0, 100.0, 2.0, 20.0, 100.5, 5.]

    Kp[6:12] = Kp[0:6]
    Kd[6:12] = Kd[0:6]

    Kp[12] = 150.0
    Kd[12] = 10.0

    Kp[13:17] = Kp[18:22] = 20.0
    Kp[17] = Kp[22] = 11.0

    Kd[13:17] = Kd[18:22] = 5.0
    Kd[17] = Kd[22] = 0.1

    # Compute feedforward torque
    tau_ff = 0

    tau = tau_ff + 1.5*scale*Kp * (q_desired - q_current) + scale/3*Kd * (dq_desired - dq_current)
    data.ctrl = tau


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


class G1MujocoSimulation:
    def __init__(self):
        rospy.init_node("g1_mujoco_sim", anonymous=True)

        # Find the model path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("g1_mujoco_sim")
        model_path = os.path.join(pkg_path, "..", "g1_description", "g1_23dof.xml")

        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.data.qpos = q_init
        # q_corrected = self.data.qpos.copy()
        # q_corrected[2] = self.data.qpos[2]
        # self.data.qpos = q_corrected

        # Real-time settings
        self.sim_timestep = self.model.opt.timestep
        self.real_time_factor = 1. # 1.0 = real time

        # Create viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Running flag
        self.running = True

        joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ]

        contact_frame_names = [
            "left_foot_upper_right",
            "left_foot_lower_right",
            "left_foot_upper_left",
            "left_foot_lower_left",
            "right_foot_upper_right",
            "right_foot_lower_right",
            "right_foot_upper_left",
            "right_foot_lower_left",
        ]

    def run(self):
        """Run simple real-time simulation."""
        prev_time = time.time()
        sim_time = 0.0

        try:
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
                    PID_controller(self.data, q_init)
                    mujoco.mj_step(self.model, self.data)
                    sim_time += self.sim_timestep

                # Render the current state
                self.viewer.render()

        except Exception as e:
            rospy.logerr(f"Simulation error: {e}")
        finally:
            if self.viewer:
                self.viewer.close()


if __name__ == "__main__":
    # Run simulation
    sim = G1MujocoSimulation()
    sim.run()

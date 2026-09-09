# One single-rigid-body state along a trajectory.
#
# Matches the MPC state vector:
#   [roll pitch yaw | x y z | wx wy wz | vx vy vz | g]
# with the orientation as ZYX Euler angles of the base and the position and
# linear velocity of the CoM, all in the world frame. Yaw is unwrapped, not
# folded into [-pi, pi], so a turn does not produce a spurious 2*pi error.

# Index of this state within the horizon
int32 trajectory_index

# Base orientation as roll, pitch, yaw [rad]
geometry_msgs/Vector3 orientation

# CoM position [m]
geometry_msgs/Vector3 position

# Base angular velocity, world frame [rad/s]
geometry_msgs/Vector3 angular_velocity

# CoM linear velocity, world frame [m/s]
geometry_msgs/Vector3 linear_velocity

# Gravity augmentation, constant [m/s^2]
float64 gravity

# This message is for the states of a Single Rigid Body
# orientation of the torso, position of the CoM, angular_velocity of the torso, velocity of the CoM

# Trajectory index
int32 trajectory_index

# Base orientation in Euler angles (roll, pitch, yaw) [rad]
geometry_msgs/Vector3 orientation

# CoM position (x, y, z) [m]
geometry_msgs/Vector3 position

# Base angular velocity (wx, wy, wz) [rad/s]
geometry_msgs/Vector3 angular_velocity

# CoM linear velocity (vx, vy, vz) [m/s]
geometry_msgs/Vector3 linear_velocity

# Gravity [m/s^2]
float32 gravity
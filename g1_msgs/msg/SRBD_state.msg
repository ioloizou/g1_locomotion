# This message is for the states of a Single Rigid Body
# orientation of the torso, position of the CoM, angular_velocity of the torso, velocity of the CoM

# Header with timestamp and frame information
std_msgs/Header header

# Base orientation in Euler angles (roll, pitch, yaw) [rad]
geometry_msgs/Vector3 orientation

# CoM position (x, y, z) [m]
geometry_msgs/Vector3 position

# Base angular velocity (wx, wy, wz) [rad/s]
geometry_msgs/Vector3 angular_velocity

# CoM linear velocity (vx, vy, vz) [m/s]
geometry_msgs/Vector3 velocity



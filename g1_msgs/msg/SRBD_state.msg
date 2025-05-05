# This message is for the states, contant points, and forces of a Single Rigid Body

# Header with timestamp and frame information
std_msgs/Header header

# State for the whole horizon
State[] states_horizon

# Array of contact points with their corresponding information
ContactPoint[] contacts

# Landing position of swinging leg (Point Contact used Not Line contact)
geometry_msgs/Point landing_position
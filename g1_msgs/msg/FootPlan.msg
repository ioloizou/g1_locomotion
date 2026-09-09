# Planned gait state of one foot, produced by the MPC node's gait scheduler.
#
# The simulation follows this instead of running its own swing clock: two
# independent timers drift apart, and a foot that the plan believes is in
# stance while the simulation is still swinging it is how the robot falls.

# "left" or "right"
string name

# True while the foot is scheduled to carry load.
bool in_contact

# Progress through the current swing, in [0, 1). Zero while in stance.
float64 swing_phase

# Where this foot is planned to touch down, world frame. Refers to the foot
# (point-contact) frame, the same frame the swing task tracks.
geometry_msgs/Point landing_position

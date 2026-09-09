# One contact point of the single-rigid-body model.
#
# The four points are always ordered L-heel, L-toe, R-heel, R-toe.

# Frame name of the contact point
string name

# Position, world frame [m]
geometry_msgs/Vector3 position

# Ground reaction force, world frame [N]
geometry_msgs/Vector3 force

# True when this point is scheduled to carry load
bool active

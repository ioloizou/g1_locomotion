# Single-rigid-body state, contact points and forces.
#
# Published by the simulation as the measured state (`/srbd_current`, one entry
# in states_horizon) and by the MPC as the solution (`/mpc_solution`).

# Header with timestamp and frame information
std_msgs/Header header

# The MPC's optimal rollout: what it predicts will happen under the forces
# below. Starts at the measured state, so it is a prediction, not a target.
State[] states_horizon

# The trajectory the MPC is regulating towards: level trunk, commanded heading
# and velocity, nominal CoM height.
#
# This is what the whole-body controller tracks. The rollout above cannot serve
# that purpose: it begins at the measurement, so one step ahead it is still
# essentially the measurement and leaves the tracking tasks with no error to
# correct. Empty on the measured-state topic.
State[] reference_horizon

# Contact points, always ordered L-heel, L-toe, R-heel, R-toe.
ContactPoint[] contacts

# Per-foot gait plan: contact flag, swing phase and landing target.
# Ordered left, right. Empty on the measured-state topic.
FootPlan[] foot_plan

# True when the MPC's QP solved; false means the forces are a static fallback.
bool solver_ok

"""Nominal joint configuration of the G1, in the xbot2/pinocchio convention.

Layout: ``[base_xyz (3), base_quat_xyzw (4), joints (23)]``.

The base height here is only a spawn guess; the height that actually places the
soles on the ground is derived from forward kinematics by
``scripts/extract_robot_params.py`` and stored as ``robot/base_height`` in the
config, which is what the simulation uses.
"""

import numpy as np

# Joint names in model order, so the ordering assumption can be asserted at
# start-up instead of silently going wrong when the URDF changes.
JOINT_NAMES = [
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

# Crouched stance: the knees must not be straight or the leg Jacobian loses
# rank in the vertical direction and the whole-body QP cannot push against the
# ground.  These are the angles the stack was tuned around.
JOINT_ANGLES = {
    "left_hip_pitch_joint": -0.60,
    "left_knee_joint": 1.20,
    "left_ankle_pitch_joint": -0.60,
    "right_hip_pitch_joint": -0.60,
    "right_knee_joint": 1.20,
    "right_ankle_pitch_joint": -0.60,
}

BASE_SPAWN_HEIGHT = 0.68


def nominal_q():
    """Assemble the nominal configuration vector."""
    q = np.zeros(7 + len(JOINT_NAMES))
    q[2] = BASE_SPAWN_HEIGHT
    q[6] = 1.0  # quaternion w, xbot stores (x, y, z, w)
    for name, value in JOINT_ANGLES.items():
        q[7 + JOINT_NAMES.index(name)] = value
    return q


Q_NOMINAL = nominal_q()

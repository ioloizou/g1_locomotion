#!/usr/bin/env python3
"""Standalone MuJoCo check: hold the nominal posture with joint PD only.

No ROS master, no MPC, no whole-body QP.  Useful for telling "the model, the
contacts or the spawn height are wrong" apart from "the controller is wrong",
which is otherwise hard to do once everything is running together.

    rosrun g1_mujoco_sim pd_stand_demo.py --headless --duration 3
"""

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))
from g1_mujoco_sim.nominal_configuration import JOINT_NAMES, Q_NOMINAL  # noqa: E402

# Per-joint gains for the leg chain, reused for both legs, then waist and arms.
LEG_KP = [530.0, 570.0, 550.0, 270.0, 130.0, 30.0]
LEG_KD = [60.0, 100.0, 2.0, 20.0, 100.0, 5.0]


def gains(num_joints):
    kp = np.zeros(num_joints)
    kd = np.zeros(num_joints)
    kp[0:6] = kp[6:12] = LEG_KP
    kd[0:6] = kd[6:12] = LEG_KD
    kp[12] = 150.0
    kd[12] = 10.0
    kp[13:] = 20.0
    kd[13:] = 5.0
    return kp, kd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--base-height", type=float, default=0.675673,
                        help="pelvis spawn height, robot/base_height in the config")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, os.pardir, os.pardir, "g1_description",
                              "g1_23dof.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.001

    q = Q_NOMINAL.copy()
    data.qpos[0:3] = [0.0, 0.0, args.base_height]
    data.qpos[3] = q[6]
    data.qpos[4:7] = q[3:6]
    data.qpos[7:] = q[7:]
    mujoco.mj_forward(model, data)

    q_desired = np.array(q[7:])
    kp, kd = gains(len(JOINT_NAMES))

    viewer = None
    if not args.headless:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data)

    steps = int(args.duration / model.opt.timestep)
    try:
        for step in range(steps):
            # Gravity compensation plus PD. Without the feed-forward term the
            # legs sag by a couple of centimetres and the test says more about
            # the gains than about the model.
            data.ctrl[:] = (data.qfrc_bias[6:]
                            + kp * (q_desired - data.qpos[7:])
                            - kd * data.qvel[6:])
            mujoco.mj_step(model, data)
            if viewer is not None and step % 20 == 0:
                if not viewer.is_alive:
                    break
                viewer.render()
    finally:
        if viewer is not None:
            viewer.close()

    print("pelvis height after %.2f s: %.4f m" % (args.duration, data.qpos[2]))


if __name__ == "__main__":
    main()

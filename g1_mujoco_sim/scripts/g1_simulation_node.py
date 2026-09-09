#!/usr/bin/env python3
"""ROS node: MuJoCo simulation of the G1 driven by the MPC plan."""

import argparse
import os
import sys

import rospkg
import rospy

from g1_mujoco_sim.nominal_configuration import JOINT_NAMES, Q_NOMINAL
from g1_mujoco_sim.simulation import G1Simulation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true",
                       help="run without opening the MuJoCo viewer")
    parser.add_argument("--duration", type=float, default=None,
                       help="stop after this many seconds of simulated time")
    parser.add_argument("--free-run", action="store_true",
                       help="ignore simulation/realtime and run as fast as possible")
    # ROS injects __name:= and __log:= style arguments.
    known, _ = parser.parse_known_args(rospy.myargv(sys.argv)[1:])
    return known


def main():
    args = parse_args()
    rospy.init_node("g1_mujoco_sim")

    description = rospkg.RosPack().get_path("g1_description")
    urdf_path = os.path.join(description, "g1_23dof.urdf")
    model_path = os.path.join(description, "g1_23dof.xml")

    headless = args.headless or rospy.get_param("~headless", False)
    duration = args.duration
    if duration is None:
        duration = rospy.get_param("~duration", 0.0) or None

    simulation = G1Simulation(urdf_path, model_path, Q_NOMINAL, JOINT_NAMES,
                              headless=headless, duration=duration)
    if args.free_run:
        rospy.set_param("simulation/realtime", False)

    rospy.loginfo("g1_mujoco_sim: running (headless=%s, duration=%s)",
                  headless, duration)
    try:
        simulation.run()
    finally:
        simulation.close()

    if simulation.fallen:
        sys.exit(2)


if __name__ == "__main__":
    main()

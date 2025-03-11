cmake_minimum_required(VERSION 3.0.2)
project(g1_mujoco_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  tf
  tf2_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/run_simulation.py
  scripts/ros_run_simulation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  src
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

cmake_minimum_required(VERSION 3.10)
project(g1_mujoco_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roslaunch
  std_msgs
  sensor_msgs
  geometry_msgs
  rosgraph_msgs
  visualization_msgs
  tf
  g1_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs rosgraph_msgs
                 visualization_msgs tf g1_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/g1_simulation_node.py
  scripts/extract_robot_params.py
  scripts/pd_stand_demo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# The Python package is installed directly rather than through
# catkin_python_setup(): catkin's distutils wrapper passes --install-layout,
# which setuptools removed, so it fails on any recent system.
install(DIRECTORY src/g1_mujoco_sim
  DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  roslaunch_add_file_check(launch)
endif()

cmake_minimum_required(VERSION 3.10)
project(g1_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
  ContactPoint.msg
  Feet_reference.msg
  FootPlan.msg
  SRBD_state.msg
  State.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)

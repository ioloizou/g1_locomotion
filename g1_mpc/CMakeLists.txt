cmake_minimum_required(VERSION 3.10)
project(g1_mpc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    rosgraph_msgs
    g1_msgs
)

find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(yaml-cpp REQUIRED)

# The condensed QP is built with fixed-size Eigen blocks; lift the stack
# allocation limit rather than forcing every temporary onto the heap.
add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=0)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES g1_mpc
    CATKIN_DEPENDS roscpp std_msgs geometry_msgs rosgraph_msgs g1_msgs
    DEPENDS EIGEN3
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

add_library(g1_mpc
    src/mpc.cpp
    src/gait_scheduler.cpp
    src/footstep_planner.cpp
    src/parameters.cpp
)
add_dependencies(g1_mpc ${catkin_EXPORTED_TARGETS})
target_link_libraries(g1_mpc
    ${catkin_LIBRARIES}
    OsqpEigen::OsqpEigen
    yaml-cpp
)

add_executable(mpc_node src/mpc_node.cpp)
add_dependencies(mpc_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(mpc_node
    g1_mpc
    ${catkin_LIBRARIES}
)

add_executable(mpc_selftest test/mpc_selftest.cpp)
add_dependencies(mpc_selftest ${catkin_EXPORTED_TARGETS})
target_link_libraries(mpc_selftest g1_mpc ${catkin_LIBRARIES})

install(TARGETS g1_mpc mpc_node mpc_selftest
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp"
)

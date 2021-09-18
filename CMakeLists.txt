# What version of CMake is needed?
cmake_minimum_required(VERSION 2.8.3)

# The name of this package.
project(mpc_pkg)

# Find the catkin build system, and any other packages on which we depend.
find_package(catkin REQUIRED COMPONENTS
    roscpp
    geometry_msgs
    std_msgs
    message_generation
    actionlib
    actionlib_msgs
    trajectory_msgs
    control_msgs
    sensor_msgs
    tf2
    tf2_ros
)
find_package(casadi)
find_package(Eigen3 REQUIRED)

## Generate MESSAGES in the 'msg' folder
add_message_files(
    FILES
    TwoPoses.msg
    TwoPoints.msg
    ClothMesh.msg
    OptiData.msg
    HorizonControls.msg
)

## Generate services in the 'srv' folder
#add_service_files(
#   FILES
   #CheckInitialPositions.srv
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   actionlib_msgs
   trajectory_msgs
   control_msgs
   sensor_msgs
   geometry_msgs
 )

# Declare our catkin package.
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES demopkg
   CATKIN_DEPENDS roscpp std_msgs message_runtime
#  DEPENDS system_lib
)

# Specify locations of header files.
include_directories(../include)
include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(EIGEN3_INCLUDE_DIR)

# Create executable.
file(GLOB MPC_CL_SOURCES "src/*_functions.cpp" "src/mpc_cl_node.cpp")
add_executable(mpc_cl_node ${MPC_CL_SOURCES})
file(GLOB MPC_RT_SOURCES "src/*_functions.cpp" "src/mpc_rt_node.cpp")
add_executable(mpc_rt_node ${MPC_RT_SOURCES})
file(GLOB MPC_OPTI_SOURCES "src/*_functions.cpp" "src/mpc_opti_node.cpp")
add_executable(mpc_opti_node ${MPC_OPTI_SOURCES})
file(GLOB READ_SOURCES "src/general_functions.cpp" "src/read_node.cpp")
add_executable(read_node ${READ_SOURCES})

# Link necessary libraries.
target_link_libraries(mpc_cl_node ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
target_link_libraries(mpc_cl_node casadi)
target_link_libraries(mpc_rt_node ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
target_link_libraries(mpc_rt_node casadi)
target_link_libraries(mpc_opti_node ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
target_link_libraries(mpc_opti_node casadi)
target_link_libraries(read_node ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

add_dependencies(mpc_cl_node ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(mpc_rt_node ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(mpc_opti_node ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(read_node ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})

# Set flag: C++11 needed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")









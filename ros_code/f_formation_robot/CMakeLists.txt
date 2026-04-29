cmake_minimum_required(VERSION 3.8)
project(f_formation_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(spatial_utils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(f_formation_robot
  src/FollowerRobotNode.cpp
  src/MoveToTarget.cpp
  src/f_formation_robot.cpp
)

ament_target_dependencies(f_formation_robot
  cv_bridge
  geometry_msgs
  nav2_msgs
  OpenCV
  rclcpp
  rclcpp_action
  sensor_msgs
  spatial_utils
  std_msgs
  tf2_ros
)

target_link_libraries(f_formation_robot
  ${OpenCV_LIBS}
)

install(TARGETS
  f_formation_robot
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/f_formation_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

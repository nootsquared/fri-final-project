cmake_minimum_required(VERSION 3.5)
project(nav_goals)

# Required dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)  # For frame transformations
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Add executable
add_executable(send_goal src/send_goal.cpp)
ament_target_dependencies(send_goal
  rclcpp
  rclcpp_action
  nav2_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Install executable
install(TARGETS
  send_goal
  DESTINATION lib/${PROJECT_NAME}
)

# Export package
ament_package()
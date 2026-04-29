cmake_minimum_required(VERSION 3.8)
project(spatial_transforms)

# Find dependencies

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(spatial_utils REQUIRED)

# Find GTK+ (using PkgConfig)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# Include directories
include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${GTK3_INCLUDE_DIRS}
  # include
)

# Link directories
link_directories(${GTK3_LIBRARY_DIRS})

# Define the executable
add_executable(spatial_transform_node src/spatial_transform_node.cpp)
target_link_libraries(spatial_transform_node ${GTK3_LIBRARIES})

# Add dependencies to the executable
ament_target_dependencies(spatial_transform_node
  rclcpp
  spatial_utils
  tf2_ros
  geometry_msgs
  Eigen3
)

add_executable(tf_listener_node src/tf_listener_node.cpp)
ament_target_dependencies(tf_listener_node
  spatial_utils
  rclcpp
  tf2_ros
  geometry_msgs
  Eigen3
)

# Export GTK-related settings
add_definitions(${GTK3_CFLAGS_OTHER})

# Install the executable
install(
  TARGETS
  spatial_transform_node
  tf_listener_node
  DESTINATION lib/${PROJECT_NAME}
)

# install(
#   DIRECTORY include/
#   DESTINATION include
# )

# Export package dependencies
ament_export_dependencies(rclcpp tf2_ros geometry_msgs Eigen3)
ament_export_include_directories(include)

# Register the package
ament_package()

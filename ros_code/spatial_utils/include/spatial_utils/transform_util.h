#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <string>

/*
You will write all 3 of these.
*/

//Take the TransformStamped. Output a 4x4 rigid transformation.
Eigen::MatrixXd transformToMatrix(const geometry_msgs::msg::TransformStamped &transform);

//Take a 4x4 rigid transformation, output a TransformStamped
geometry_msgs::msg::TransformStamped matrixToTransform(
    const Eigen::MatrixXd &matrix, const std::string &parent_frame, const std::string &child_frame);

//Print the contents of a TransformStamped
void printTransform(const rclcpp::Logger &logger, geometry_msgs::msg::TransformStamped &transform);
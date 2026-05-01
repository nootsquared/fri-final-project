#include "f_formation_robot/FollowerRobotNode.h"
#include <spatial_utils/transform_util.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

using namespace std;

FollowerRobotNode::FollowerRobotNode(
    double follow_distance,
    double angle_threshold)
:   Node("f_formation_robot_node"),
    follow_distance_(follow_distance),
    angle_threshold_(angle_threshold),
    move_to_target_(this),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_),
    tf_broadcaster_(this),
    m_map_to_go_to_(Eigen::Matrix4d::Identity()),
    fformation_detected_(false),
    goal_angle_(0.0),
    goal_distance_(0.0),
    goal_angle_prev_(999.0),
    entry_facing_(0.0)
{
    detected_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/fformation/detected", 1,
        std::bind(&FollowerRobotNode::fformationDetectedCallback, this,
                  std::placeholders::_1));

    angle_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/fformation/goal_angle", 1,
        std::bind(&FollowerRobotNode::goalAngleCallback, this,
                  std::placeholders::_1));

    distance_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/fformation/goal_distance", 1,
        std::bind(&FollowerRobotNode::goalDistanceCallback, this,
                  std::placeholders::_1));

    facing_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/fformation/entry_facing", 1,
        std::bind(&FollowerRobotNode::entryFacingCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
        "F-formation follower ready. follow_distance=%.2f m", follow_distance_);
}

FollowerRobotNode::~FollowerRobotNode() {}

void FollowerRobotNode::fformationDetectedCallback(
    const std_msgs::msg::Bool::SharedPtr msg)
{
    fformation_detected_ = msg->data;
    if (!fformation_detected_) {
        RCLCPP_INFO_STREAM(this->get_logger(), "F-formation lost — standing by.");
        goal_angle_prev_ = 999.0;
    }
}

void FollowerRobotNode::goalDistanceCallback(
    const std_msgs::msg::Float32::SharedPtr msg)
{
    goal_distance_ = static_cast<double>(msg->data);
}

void FollowerRobotNode::entryFacingCallback(
    const std_msgs::msg::Float32::SharedPtr msg)
{
    entry_facing_ = static_cast<double>(msg->data);
}

void FollowerRobotNode::goalAngleCallback(
    const std_msgs::msg::Float32::SharedPtr msg)
{
    goal_angle_ = static_cast<double>(msg->data);

    if (fformation_detected_) {
        computeAndAct();
    }
}

Eigen::MatrixXd FollowerRobotNode::computeGoToFromFFormation(
    double angle, double distance)
{
    double tx = distance * cos(angle);
    double ty = -distance * sin(angle);
    double tz = 0.0;

    double magnitude = sqrt(tx*tx + ty*ty + tz*tz);

    Eigen::AngleAxisd rot_z(-entry_facing_, Eigen::Vector3d::UnitZ());
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(4, 4);
    matrix.block<3,3>(0,0) = rot_z.toRotationMatrix();

    double unit_x = tx / magnitude;
    double unit_y = ty / magnitude;
    double scalar  = magnitude - follow_distance_;

    matrix(0, 3) = unit_x * scalar;
    matrix(1, 3) = unit_y * scalar;
    matrix(2, 3) = 0.0;

    return matrix;
}

void FollowerRobotNode::computeAndAct()
{
    try {
        geometry_msgs::msg::TransformStamped map_to_base_link =
            tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);

        double angle_delta = fabs(goal_angle_ - goal_angle_prev_);
        bool target_moved  = angle_delta > angle_threshold_;

        if (target_moved) {
            RCLCPP_INFO_STREAM(this->get_logger(),
                "Entry point updated — angle=" << goal_angle_
                << " rad, dist=" << goal_distance_ << " m");

            if (goal_distance_ > follow_distance_) {
                Eigen::MatrixXd m_map_to_base_link =
                    transformToMatrix(map_to_base_link);

                Eigen::MatrixXd m_go_to =
                    computeGoToFromFFormation(goal_angle_, goal_distance_);

                m_map_to_go_to_ = m_map_to_base_link * m_go_to;

                move_to_target_.copyToGoalPoseAndSend(m_go_to);

                goal_angle_prev_ = goal_angle_;
            }
        }

        geometry_msgs::msg::TransformStamped tf1 =
            matrixToTransform(m_map_to_go_to_, "map", "entry_point");
        tf1.header.stamp = this->get_clock()->now();
        tf_broadcaster_.sendTransform(tf1);

    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
    }
}

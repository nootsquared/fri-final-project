#ifndef FOLLOWER_ROBOT_NODE_H
#define FOLLOWER_ROBOT_NODE_H

#include "f_formation_robot/MoveToTarget.h"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Dense>

class FollowerRobotNode : public rclcpp::Node {
public:
    FollowerRobotNode(
        double follow_distance = 0.1,
        double angle_threshold = 0.08);
    ~FollowerRobotNode();

protected:
    void fformationDetectedCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void goalAngleCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void entryFacingCallback(const std_msgs::msg::Float32::SharedPtr msg);

    Eigen::MatrixXd computeGoToFromFFormation(double angle, double distance);
    void computeAndAct();

    double follow_distance_;
    double angle_threshold_;

    MoveToTarget move_to_target_;

    tf2_ros::Buffer               tf_buffer_;
    tf2_ros::TransformListener    tf_listener_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    Eigen::MatrixXd m_map_to_go_to_;

    bool   fformation_detected_;
    double goal_angle_;
    double goal_distance_;
    double goal_angle_prev_;
    double entry_facing_;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr    detected_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr angle_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr distance_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr facing_sub_;
};

#endif

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

/*
    F-Formation follower node.

    Adapted from the AprilTag follower template.  Instead of subscribing to
    AprilTag detections and looking up TF frames for the tag, this node
    subscribes to the Python perception node's output:

        /fformation/detected      std_msgs/Bool
        /fformation/goal_angle    std_msgs/Float32   (radians, + = right)
        /fformation/goal_distance std_msgs/Float32   (metres to entry point)

    The goal angle and distance describe the F-formation entry point in the
    robot's forward-looking camera frame, which is aligned with base_link's
    +x axis.  The rest of the Nav2 goal-sending machinery (MoveToTarget,
    TF broadcasting, matrix math) is kept exactly as in the original template.
*/
class FollowerRobotNode : public rclcpp::Node {
public:
    /*
        follow_distance: how close to stop to the entry point (metres).
        angle_threshold: minimum change in goal angle (radians) before
                         re-sending a navigation goal.  Prevents spamming
                         Nav2 when the formation is stable.
    */
    FollowerRobotNode(
        double follow_distance   = 0.1,
        double angle_threshold   = 0.08);
    ~FollowerRobotNode();

protected:
    // Callbacks — one per subscribed topic.
    void fformationDetectedCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void goalAngleCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void entryFacingCallback(const std_msgs::msg::Float32::SharedPtr msg);

    /*
        F-formation equivalent of the template's computeGoToFrameFromBaseLink.

        Given the goal angle (bearing from robot forward axis to entry point,
        positive = right) and distance (metres), returns a 4x4 rigid
        transformation in base_link that describes where the robot should go
        and the heading it should hold when it arrives.

        The math mirrors the original exactly — only the input source changes.
    */
    Eigen::MatrixXd computeGoToFromFFormation(double angle, double distance);

    /*
        Decides whether a new Nav2 goal should be sent, computes it, and
        broadcasts the goal frame via TF — exactly as in the original template.
    */
    void computeAndAct();

    // ---- Parameters ----
    double follow_distance_;   // stop this far from the entry point
    double angle_threshold_;   // minimum angle change to trigger a new goal

    // ---- Nav2 goal sender (unchanged from template) ----
    MoveToTarget move_to_target_;

    // ---- TF2 (unchanged from template) ----
    tf2_ros::Buffer              tf_buffer_;
    tf2_ros::TransformListener   tf_listener_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // ---- Persistent goal transform (unchanged from template) ----
    Eigen::MatrixXd m_map_to_go_to_;

    // ---- F-formation state ----
    bool   fformation_detected_;
    double goal_angle_;        // radians, + = right of camera axis
    double goal_distance_;     // metres to entry point
    double goal_angle_prev_;   // last angle for which a goal was sent
    double entry_facing_;      // heading to hold on arrival (toward o-space centre)

    // ---- Subscribers ----
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr    detected_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr angle_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr distance_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr facing_sub_;
};

#endif

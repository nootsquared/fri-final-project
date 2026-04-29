#ifndef MOVE_TO_TARGET_H
#define MOVE_TO_TARGET_H

#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <Eigen/Dense>
#include <mutex>

/*
    I wrapped all of the functionality of sending navigation goals into this
        class.
    The idea is to simplify your homework by only requiring you to implemnent
        part of the solution, but to provide you with a reference to the rest
        of the solution.
*/
class MoveToTarget {
public:
    /*
        This class requires an rclcpp::Node, but I just use the this pointer
            from FollowerRobotNode.
    */
    MoveToTarget(rclcpp::Node *node);
    ~MoveToTarget();
    
    //I provide both callbacks and set up send_goal_options for you.
    void goal_response_callback(
        std::shared_ptr<rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>> goal_handle);
    void result_callback(
        const rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>::WrappedResult &result);

    //You will implement this.
    void copyToGoalPoseAndSend(Eigen::MatrixXd &goal_pose_relative_to_base_link);

protected:
    /*
        I used these exact class attributes, though you are at liberty to
            change them.
    */
    rclcpp::Node *node_;
    rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr client_;
    rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions send_goal_options_;
};

#endif
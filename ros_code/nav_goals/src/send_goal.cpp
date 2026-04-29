#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <functional>

// void goal_response_callback(std::shared_ptr<rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>> goal_handle) {
//     if (!goal_handle) {
//         RCLCPP_ERROR(rclcpp::get_logger("nav_goal_sender"), "Goal was rejected!");
//     } else {
//         RCLCPP_INFO(rclcpp::get_logger("nav_goal_sender"), "Goal accepted!");
//     }
// }

class NavGoalSender : public rclcpp::Node {
public:
    NavGoalSender() :
        Node("nav_goal_sender"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_)
    {
        // this->declare_parameter("use_sim_time", true);
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));

        client_ = rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(this, "navigate_to_pose");

        while (!client_->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_INFO(this->get_logger(), "Waiting for Nav2 action server...");
        }
    }
    
    void goal_response_callback(std::shared_ptr<rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>> goal_handle) {
        if (!goal_handle) {
            RCLCPP_ERROR(rclcpp::get_logger("nav_goal_sender"), "Goal was rejected!");
        } else {
            RCLCPP_INFO(rclcpp::get_logger("nav_goal_sender"), "Goal accepted!");
        }
    }    

    /*
    void set_goal_pose(geometry_msgs::msg::PoseStamped &goal_pose, double x, double y) {
    goal_pose.header.frame_id = "base_link";
    goal_pose.header.stamp = rclcpp::Clock().now();
    goal_pose.pose.position.x = x;
    goal_pose.pose.position.y = y;

    double yaw = atan2(y, x);

    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    goal_pose.pose.orientation.x = q.x();
    goal_pose.pose.orientation.y = q.y();
    goal_pose.pose.orientation.z = q.z();
    goal_pose.pose.orientation.w = q.w();
}
    */

    bool send_goal() {
        geometry_msgs::msg::PoseStamped goal_pose;
        goal_pose.header.frame_id = "base_link";
        goal_pose.header.stamp = this->get_clock()->now();
        goal_pose.pose.position.x = 1.0; // Move 2 meters forward
        goal_pose.pose.position.y = 0.0; // No lateral movement
        goal_pose.pose.orientation.w = 1.0; // No rotation

        bool lookup_good = false;
        geometry_msgs::msg::PoseStamped global_goal;
        try {
            geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
            if(tf_buffer_.canTransform("map", "base_link", transform.header.stamp, tf2::durationFromSec(2.0))) {
                RCLCPP_INFO(this->get_logger(), "Transform is good!");
                tf2::doTransform(goal_pose, global_goal, transform);
                lookup_good = true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Transform is bad!");
            }
            

            nav2_msgs::action::NavigateToPose::Goal goal_msg = nav2_msgs::action::NavigateToPose::Goal();
            goal_msg.pose = global_goal;

            RCLCPP_INFO(this->get_logger(), "Sending goal...");

            rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions send_goal_options =
                rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
            // send_goal_options.goal_response_callback = goal_response_callback;
            send_goal_options.goal_response_callback = std::bind(&NavGoalSender::goal_response_callback, this, std::placeholders::_1);

            client_->async_send_goal(goal_msg, send_goal_options);
            
         } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Could not transform goal to map frame: %s", ex.what());
            lookup_good = false;
        }
        
        return lookup_good;
    }

private:
    rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr client_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    std::shared_ptr<NavGoalSender> node = std::make_shared<NavGoalSender>();
    rclcpp::Rate loop_rate(2.0); // 1 Hz retry loop
    bool keep_going = true;
    while (rclcpp::ok() && keep_going) {
        keep_going = !node->send_goal(); // Keep trying to send goal until successful
        rclcpp::spin_some(node); // Process callbacks
        loop_rate.sleep();
    }
    rclcpp::shutdown();
    return 0;
}

#include <f_formation_robot/FollowerRobotNode.h>
#include <rclcpp/rclcpp.hpp>

/*
    I provide this code. This file needs no modifications for your homework.
*/
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    std::shared_ptr<FollowerRobotNode> node = std::make_shared<FollowerRobotNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

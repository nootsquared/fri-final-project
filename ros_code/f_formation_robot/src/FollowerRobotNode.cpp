#include "f_formation_robot/FollowerRobotNode.h"
#include <spatial_utils/transform_util.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

using namespace std;

// ---------------------------------------------------------------------------
// Constructor — same initializer list structure as the original template.
// Subscribers swapped: AprilTag → F-formation topics.
// ---------------------------------------------------------------------------
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
    goal_angle_prev_(999.0),  // sentinel — forces first goal to be sent
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

// ---------------------------------------------------------------------------
// Subscriber callbacks
// ---------------------------------------------------------------------------

void FollowerRobotNode::fformationDetectedCallback(
    const std_msgs::msg::Bool::SharedPtr msg)
{
    fformation_detected_ = msg->data;
    if (!fformation_detected_) {
        RCLCPP_INFO_STREAM(this->get_logger(), "F-formation lost — standing by.");
        goal_angle_prev_ = 999.0;   // reset so next detection sends a fresh goal
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

/*
    Called every time the Python node publishes a new goal angle.
    This is the main trigger — mirrors aprilTagCallback from the template,
    which called computeAndAct() whenever a tag was detected.
*/
void FollowerRobotNode::goalAngleCallback(
    const std_msgs::msg::Float32::SharedPtr msg)
{
    goal_angle_ = static_cast<double>(msg->data);

    if (fformation_detected_) {
        computeAndAct();
    }
}

// ---------------------------------------------------------------------------
// computeGoToFromFFormation
//
// Direct replacement for computeGoToFrameFromBaseLink from the template.
// The logic is identical — compute a 4x4 rigid transform in base_link that
// places the robot follow_distance_ short of the target, facing it.
//
// Input mapping from Python perception output → base_link frame:
//   Camera forward (+z) = robot forward (+x in base_link)
//   Camera right  (+x) = robot right   (-y in base_link, ROS convention)
//
//   entry_x (base_link) =  distance * cos(angle)
//   entry_y (base_link) = -distance * sin(angle)   (sign flip: right = -y)
// ---------------------------------------------------------------------------
Eigen::MatrixXd FollowerRobotNode::computeGoToFromFFormation(
    double angle, double distance)
{
    // Entry point expressed in base_link
    double tx = distance * cos(angle);
    double ty = -distance * sin(angle);
    double tz = 0.0;   // flat floor — same assumption as the original

    double magnitude = sqrt(tx*tx + ty*ty + tz*tz);

    // Heading: face toward the o-space centre on arrival.
    // entry_facing_ is in goal_angle convention (atan2(cam_x, cam_z));
    // base_link heading = -entry_facing_ due to the ROS right-hand rule.
    Eigen::AngleAxisd rot_z(-entry_facing_, Eigen::Vector3d::UnitZ());
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(4, 4);
    matrix.block<3,3>(0,0) = rot_z.toRotationMatrix();

    // Stop follow_distance_ before the entry point (same as original)
    double unit_x = tx / magnitude;
    double unit_y = ty / magnitude;
    double scalar  = magnitude - follow_distance_;

    matrix(0, 3) = unit_x * scalar;
    matrix(1, 3) = unit_y * scalar;
    matrix(2, 3) = 0.0;

    return matrix;
}

// ---------------------------------------------------------------------------
// computeAndAct
//
// Same structure as the original template's computeAndAct:
//   1. Look up map → base_link (for broadcasting the goal as a TF frame).
//   2. Check if the goal has "moved" enough to warrant a new Nav2 goal
//      (replaces theTagMoved — we use angle change as the motion proxy).
//   3. If the entry point is far enough away: compute goal matrix, update
//      m_map_to_go_to_, and send via MoveToTarget.
//   4. Always broadcast m_map_to_go_to_ as a TF frame ("entry_point").
// ---------------------------------------------------------------------------
void FollowerRobotNode::computeAndAct()
{
    try {
        // TF lookup for map → base_link (unchanged from template)
        geometry_msgs::msg::TransformStamped map_to_base_link =
            tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);

        // "Has the target moved?" — angle-change proxy (replaces theTagMoved)
        double angle_delta = fabs(goal_angle_ - goal_angle_prev_);
        bool target_moved  = angle_delta > angle_threshold_;

        if (target_moved) {
            RCLCPP_INFO_STREAM(this->get_logger(),
                "Entry point updated — angle=" << goal_angle_
                << " rad, dist=" << goal_distance_ << " m");

            if (goal_distance_ > follow_distance_) {
                Eigen::MatrixXd m_map_to_base_link =
                    transformToMatrix(map_to_base_link);

                // Compute goal in base_link (same call pattern as original)
                Eigen::MatrixXd m_go_to =
                    computeGoToFromFFormation(goal_angle_, goal_distance_);

                // Express goal in map frame and cache (same as original)
                m_map_to_go_to_ = m_map_to_base_link * m_go_to;

                // Send Nav2 goal (unchanged from template)
                move_to_target_.copyToGoalPoseAndSend(m_go_to);

                goal_angle_prev_ = goal_angle_;
            }
        }

        // Always broadcast goal frame as TF (unchanged from template)
        geometry_msgs::msg::TransformStamped tf1 =
            matrixToTransform(m_map_to_go_to_, "map", "entry_point");
        tf1.header.stamp = this->get_clock()->now();
        tf_broadcaster_.sendTransform(tf1);

    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(),
            "TF lookup failed: %s", ex.what());
    }
}

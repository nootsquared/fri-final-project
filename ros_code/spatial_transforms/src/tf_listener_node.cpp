#include <spatial_utils/transform_util.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Dense>

class TFListenerNode : public rclcpp::Node {
public:
    TFListenerNode() :
        Node("tf_listener_node"),
        tf_buffer_(this->get_clock()),
            //tf2_ros::Buffer(std::shared_ptr<rclcpp::Clock>)
            //tf_buffer stores TF data, allowing for lookups
            //(you may need lots of transform data to do a lookup - Remember all of our TF trees)
        tf_listener_(tf_buffer_),
            //tf2_ros::TransformListener(tf2_ros::Buffer &)
            //The listener listens to the TF topics
            //It then uses this to fill the tf2_ros::Buffer
            //You do TF lookups in the buffer, because it stores the transforms
            //necessary to do the lookup
        tf_broadcaster_(this)
        {}

    Eigen::MatrixXd createOffsetFrameOne() {
        Eigen::MatrixXd transform = Eigen::MatrixXd::Identity(4, 4);
    
        // Eigen::Vector3d random_axis(1.0, 2.0, 3.0);
        // Eigen::Vector3d normalized_axis = random_axis.normalized();

        Eigen::AngleAxisd rotation = Eigen::AngleAxisd(M_PI / 3.0, Eigen::Vector3d::UnitX());

        // Insert rotation matrix into the top-left 3x3 block
        transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    
        // Insert translation into the last column
        transform(0, 3) = 3;
        transform(1, 3) = 0;
        transform(2, 3) = 0;
    
        return transform;
    }

    Eigen::MatrixXd createOffsetFrameTwo() {
        Eigen::MatrixXd transform = Eigen::MatrixXd::Identity(4, 4);
    
        Eigen::AngleAxisd rotation = Eigen::AngleAxisd(0.0 / 2.0, Eigen::Vector3d::UnitY());

        // Insert rotation matrix into the top-left 3x3 block
        transform.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    
        // Insert translation into the last column
        transform(0, 3) = 0;
        transform(1, 3) = 0;
        transform(2, 3) = 1;
    
        return transform;
    }

    void createAndPublishOffsetFrames(geometry_msgs::msg::TransformStamped &transform) {
        Eigen::MatrixXd m0 = transformToMatrix(transform);
        Eigen::MatrixXd m1 = createOffsetFrameOne();
        Eigen::MatrixXd m2 = createOffsetFrameTwo();

        Eigen::MatrixXd out1 = m0 * m1;
        Eigen::MatrixXd out2 = m0 * m1 * m2;

        geometry_msgs::msg::TransformStamped tf1 = matrixToTransform(out1, "example_frame", "off1");
        geometry_msgs::msg::TransformStamped tf2 = matrixToTransform(out2, "example_frame", "off2");

        tf1.header.stamp = this->get_clock()->now();
        tf2.header.stamp = this->get_clock()->now();

        tf_broadcaster_.sendTransform(tf1);
        tf_broadcaster_.sendTransform(tf2);
    }
    

    void lookupTransform() {
        try{
            geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(
                "world", "example_frame", tf2::TimePointZero);

            printTransform(this->get_logger(), transform);
            createAndPublishOffsetFrames(transform);
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Could not transform world -> example_frame: %s", ex.what());
        }
    }

private:
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    std::shared_ptr<TFListenerNode> node = std::make_shared<TFListenerNode>();

    while (rclcpp::ok()) {
        node->lookupTransform();
        rclcpp::spin_some(node);
    }

    rclcpp::shutdown();
    return 0;
}

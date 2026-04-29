#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Dense>
#include <gtk/gtk.h>
#include <utility>


// void printTransformStamped(rclcpp::Logger &logger, geometry_msgs::msg::TransformStamped &msg) {

//     RCLCPP_INFO_STREAM(logger,
//         "Transform: " << "\n" <<
//         "   stamp:  " << msg.header.stamp << "\n" <<
//         "   frame_id:  " << msg.header.frame_id << "\n" <<
//         "   child_frame_id:  " << msg.child_frame_id << "\n" <<
//         "   translation:  " <<  msg.transform.translation.x <<
//         "    "  << msg.transform.translation.y <<
//         "    "  << msg.transform.translation.z <<
//         "   rotation:  " <<  msg.transform.rotation.x <<
//         "    "  << msg.transform.rotation.y <<
//         "    "  << msg.transform.rotation.z <<
//         "    "  << msg.transform.rotation.w);
// }


// ===== ROS Node for Broadcasting Transform =====
class SpatialTransformNode : public rclcpp::Node {
public:
    SpatialTransformNode() : Node("spatial_transform_node") {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        q_ = 
            Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());

        timer_ = this->create_wall_timer(std::chrono::milliseconds(50),
                                         std::bind(&SpatialTransformNode::broadcast_transform, this));
    }

    void set_transform(double x, double y, double z, double rx, double ry, double rz) {
        // RCLCPP_INFO_STREAM(this->get_logger(), "set_transform:  " << x << " " << y << " " << z);
        x_ = x;
        y_ = y;
        z_ = z;
        q_ = 
            Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ());
    }

private:
    void broadcast_transform() {
        // RCLCPP_INFO_STREAM(this->get_logger(), "broadcast_transform");
        transform_msg_.header.stamp = this->now();
        transform_msg_.header.frame_id = "world";
        transform_msg_.child_frame_id = "example_frame";

        transform_msg_.transform.translation.x = x_;
        transform_msg_.transform.translation.y = y_;
        transform_msg_.transform.translation.z = z_;

        transform_msg_.transform.rotation.x = q_.x();
        transform_msg_.transform.rotation.y = q_.y();
        transform_msg_.transform.rotation.z = q_.z();
        transform_msg_.transform.rotation.w = q_.w();

        tf_broadcaster_->sendTransform(transform_msg_);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    geometry_msgs::msg::TransformStamped transform_msg_;
    Eigen::Quaterniond q_;
    double x_, y_, z_;
};

// ===== GTK GUI for Transform Control =====
class TransformGUI {
public:
    TransformGUI(SpatialTransformNode &node) : node_(node) {
        gtk_init(nullptr, nullptr);

        window_ = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_title(GTK_WINDOW(window_), "Spatial Transform Controller");
        gtk_window_set_default_size(GTK_WINDOW(window_), 400, 300);
        g_signal_connect(window_, "destroy", G_CALLBACK(gtk_main_quit), NULL);

        vbox_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
        gtk_container_add(GTK_CONTAINER(window_), vbox_);

        // Initialize transform values
        for (int i = 0; i < 6; i++) values_[i] = 0.0;

        // Create sliders
        std::pair<GtkWidget *, GtkWidget *> x = create_slider("X", -5.0, 5.0, 0.1, 0.0);
        std::pair<GtkWidget *, GtkWidget *> y = create_slider("Y", -5.0, 5.0, 0.1, 0.0);
        std::pair<GtkWidget *, GtkWidget *> z = create_slider("Z", -5.0, 5.0, 0.1, 0.0);
        std::pair<GtkWidget *, GtkWidget *> rx = create_slider("Rx", -M_PI, M_PI, 0.01, 0.0);
        std::pair<GtkWidget *, GtkWidget *> ry = create_slider("Ry", -M_PI, M_PI, 0.01, 0.0);
        std::pair<GtkWidget *, GtkWidget *> rz = create_slider("Rz", -M_PI, M_PI, 0.01, 0.0);

        // Pack sliders
        gtk_box_pack_start(GTK_BOX(vbox_), x.first, FALSE, FALSE, 5);
        gtk_box_pack_start(GTK_BOX(vbox_), y.first, FALSE, FALSE, 5);
        gtk_box_pack_start(GTK_BOX(vbox_), z.first, FALSE, FALSE, 5);
        gtk_box_pack_start(GTK_BOX(vbox_), rx.first, FALSE, FALSE, 5);
        gtk_box_pack_start(GTK_BOX(vbox_), ry.first, FALSE, FALSE, 5);
        gtk_box_pack_start(GTK_BOX(vbox_), rz.first, FALSE, FALSE, 5);

        slider_x_ = x.second;
        slider_y_ = y.second;
        slider_z_ = z.second;
        slider_rx_ = rx.second;
        slider_ry_ = ry.second;
        slider_rz_ = rz.second;

        // Show GUI
        gtk_widget_show_all(window_);

        // Run GTK main loop
        // gtk_main();
    }

private:
    std::pair<GtkWidget *, GtkWidget *> create_slider(const char *label, double min, double max, double step, double init_val) {
        GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
        GtkWidget *slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, min, max, step);
        gtk_range_set_value(GTK_RANGE(slider), init_val);

        gtk_box_pack_start(GTK_BOX(box), gtk_label_new(label), FALSE, FALSE, 5);
        gtk_box_pack_start(GTK_BOX(box), slider, TRUE, TRUE, 5);

        g_signal_connect(slider, "value-changed", G_CALLBACK(slider_changed), this);
        return std::pair<GtkWidget *, GtkWidget *>(box, slider);
    }

    static void slider_changed(GtkRange *range, gpointer user_data) {
        TransformGUI *self = static_cast<TransformGUI *>(user_data);
        self->update_transform();
    }

    void update_transform() {
        values_[0] = gtk_range_get_value(GTK_RANGE(slider_x_));
        values_[1] = gtk_range_get_value(GTK_RANGE(slider_y_));
        values_[2] = gtk_range_get_value(GTK_RANGE(slider_z_));
        values_[3] = gtk_range_get_value(GTK_RANGE(slider_rx_));
        values_[4] = gtk_range_get_value(GTK_RANGE(slider_ry_));
        values_[5] = gtk_range_get_value(GTK_RANGE(slider_rz_));

        node_.set_transform(values_[0], values_[1], values_[2], values_[3], values_[4], values_[5]);
    }

    SpatialTransformNode &node_;
    GtkWidget *window_, *vbox_;
    GtkWidget *slider_x_, *slider_y_, *slider_z_, *slider_rx_, *slider_ry_, *slider_rz_;
    double values_[6];
};

// ===== Main Function =====
int main(int argc, char **argv) {
    // Init ROS
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SpatialTransformNode>();

    // Run GTK UI
    TransformGUI gui(*node);
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        gtk_main_iteration_do(false);
    }
    rclcpp::shutdown();
    return 0;
}

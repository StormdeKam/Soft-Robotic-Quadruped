// valve_control_gui.cpp
#include <QApplication>
#include <rclcpp/rclcpp.hpp>
#include "ValveControlWindow.hpp"

int main(int argc, char *argv[]) {
    // Initialize ROS2
    rclcpp::init(argc, argv);
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("valve_control_gui_node");

    // Initialize the Qt application
    QApplication app(argc, argv);

    // Create and show the ValveControlWindow
    vtem_control_gui::ValveControlWindow window(node);
    window.show();

    // Execute the Qt application
    int result = app.exec();

    // Shutdown ROS2 when the Qt application is closed
    rclcpp::shutdown();
    return result;
}


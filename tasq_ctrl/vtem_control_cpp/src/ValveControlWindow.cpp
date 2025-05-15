#include "ValveControlWindow.hpp"
#include <QSlider>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

namespace vtem_control_gui {

ValveControlWindow::ValveControlWindow(rclcpp::Node::SharedPtr node, QWidget* parent)
    : QWidget(parent) {

    // Initialize ROS2 publishers for 12 valves (excluding 6, 7, 12 and 13)
    for (int i = 0; i < 16; ++i) {
        if (i == 6 || i == 7 || i == 12 || i == 13)  continue;  // Skip valves with indices 6, 7, 12 and 13
        valve_publishers_[i] = node->create_publisher<std_msgs::msg::Int32>(
            "valve_" + std::to_string(i) + "_pressure", rclcpp::QoS(10));
    }

    // Calculate the number of valves excluding valves 6, 7, 12 and 13
    int numValves = 16 - 4; // 16 valves minus 4 (valves 6, 7, 12 and 13)
    
    // Resize the vectors to hold the correct number of elements
    pressureSliders_.resize(numValves);
    pressureValues_.resize(numValves);

    // Create sliders, value displays for controlling valve pressures
    layout_ = new QVBoxLayout();
    int sliderIndex = 0;
    for (int i = 0; i < 16; ++i) {
        if (i == 6 || i == 7 || i == 12 || i == 13) continue;  // Skip valves for indices 6, 7, 12 and 13

        // Horizontal layout for each slider and value display
        QHBoxLayout* valveLayout = new QHBoxLayout();

        // Label for the valve
        QLabel* label = new QLabel("Valve " + QString::number(i + 1));
        valveLayout->addWidget(label);

        // Slider for the valve pressure
        pressureSliders_[sliderIndex] = new QSlider(Qt::Horizontal);
        pressureSliders_[sliderIndex]->setRange(0, 450); // Pressure range from 0 to 450 mBar
        connect(pressureSliders_[sliderIndex], &QSlider::valueChanged, this, &ValveControlWindow::onSliderValueChanged);
        valveLayout->addWidget(pressureSliders_[sliderIndex]);

        // Label to display the current pressure value
        pressureValues_[sliderIndex] = new QLabel("0 mBar");
        valveLayout->addWidget(pressureValues_[sliderIndex]);

        // Add the horizontal layout to the main layout
        layout_->addLayout(valveLayout);

        sliderIndex++;  // Increment slider index for the next non-skipped valve
    }

    setLayout(layout_);
}

ValveControlWindow::~ValveControlWindow() {}

void ValveControlWindow::onSliderValueChanged(int value) {
    int sliderIndex = 0;
    for (int i = 0; i < 16; ++i) {
        if (i == 6 || i == 7 || i == 12 || i == 13) continue;  // Skip valves for indices 6, 7, 12 and 13
        if (sender() == pressureSliders_[sliderIndex]) {
            // Update the corresponding value label
            pressureValues_[sliderIndex]->setText(QString::number(value) + " mBar");

            // Publish the value to the corresponding ROS2 topic
            auto it = valve_publishers_.find(i);
            if (it != valve_publishers_.end() && it->second) {
                std_msgs::msg::Int32 msg;
                msg.data = value;
                it->second->publish(msg);
            }
            return;  // Exit the loop after finding the matching slider
        }
        sliderIndex++;
    }
}

} // namespace vtem_control_gui


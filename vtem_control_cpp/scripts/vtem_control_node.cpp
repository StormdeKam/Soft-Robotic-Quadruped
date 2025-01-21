#include "rclcpp/rclcpp.hpp"
#include <std_msgs/msg/int32.hpp>
#include "VtemControl.hpp"
#include <unordered_map>
#include <iostream>
#include <mutex>
#include <vector>
#include <string>

namespace vtem_control {

class ValveControlNode : public rclcpp::Node {
public:
    ValveControlNode()
        : Node("vtem_control_node"), vtemControl_("192.168.4.3", "502") {

        // Connect to VTEM
        if (!vtemControl_.connect()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect to VTEM.");
            return;
        }

        // Acknowledge errors for all slots
        vtemControl_.acknowledge_errors();

        // Initialize ROS2 subscription for each valve pressure (skipping 6,7,12 and 13)
        for (int i = 0; i < 16; ++i) {
            if (i == 6 || i == 7 || i == 12 || i == 13) continue;  // Skip valves for indices 6, 7, 12 and 13
            std::string topic_name = "valve_" + std::to_string(i) + "_pressure";
            valve_subscribers_[i] = this->create_subscription<std_msgs::msg::Int32>(
                topic_name, rclcpp::QoS(10), [this, i](std::shared_ptr<std_msgs::msg::Int32> msg) {
                    this->onPressureReceived(msg, i);
                });
        }

        // Periodic logging thread
        logThread_ = std::thread([this]() {
    while (rclcpp::ok()) {
        // Log periodically without holding the lock for too long
        {
            std::lock_guard<std::mutex> lock(dataMutex_);
            for (const auto& [valveIdx, pressure] : valvePressureMap_) {
                int actualPressure = vtemControl_.get_single_pressure(valveIdx);
                RCLCPP_INFO(this->get_logger(), "Valve %d | Commanded Pressure: %d | Actual Pressure: %d mBar",
                            valveIdx, pressure, actualPressure);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(5)); // Wait before logging again
    }
});
    }

    ~ValveControlNode() {
        // Cleanup VTEM connection
        {
            std::lock_guard<std::mutex> lock(dataMutex_);
            for (const auto& [valveIdx, _] : valvePressureMap_) {
                unsigned int slotIdx = vtemControl_.get_slot_idx_from_valve_idx(valveIdx);
                vtemControl_.deactivate_pressure_regulation(slotIdx);
            }
        }
        vtemControl_.disconnect();

        // Stop logging thread
        if (logThread_.joinable()) {
            logThread_.join();
        }
    }

private:
    // Callback for valve pressure updates
    void onPressureReceived(const std_msgs::msg::Int32::SharedPtr msg, int valveIdx) {
    if (valveIdx == 6 || valveIdx == 7 || valveIdx == 12 || valveIdx == 13) return;  // Skip valves for indices 6, 7, 12 and 13

    unsigned int newPressure = msg->data;

    // Validate pressure
    if (newPressure > 450) {
        RCLCPP_WARN(this->get_logger(), "Pressure must be between 0 and 450 mBar.");
        return;
    }

    // Log message before locking the mutex
    RCLCPP_INFO(this->get_logger(), "Received pressure for valve: %d", valveIdx);

    {
        // Only lock the mutex for shared resource access, not for long operations
        std::lock_guard<std::mutex> lock(dataMutex_);

        // Check if valve is not active
        if (valvePressureMap_.find(valveIdx) == valvePressureMap_.end()) {
            RCLCPP_INFO(this->get_logger(), "Valve %d not active, activating pressure regulation", valveIdx);
            unsigned int slotIdx = vtemControl_.get_slot_idx_from_valve_idx(valveIdx);
            if (!vtemControl_.activate_pressure_regulation(slotIdx)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to activate pressure regulation for valve %d.", valveIdx);
                return;
            }
        }

        // Update the valve pressure
        vtemControl_.set_single_pressure(valveIdx, newPressure);
        valvePressureMap_[valveIdx] = newPressure;
    }
}

    // VTEM controller instance
    vtem_control::VtemControl vtemControl_;

    // ROS2 subscribers for each valve pressure
    std::unordered_map<int, rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> valve_subscribers_;

    // Map for valve pressures
    std::unordered_map<unsigned int, unsigned int> valvePressureMap_;
    std::mutex dataMutex_;

    // Logging thread for periodic pressure updates
    std::thread logThread_;
};

} // namespace vtem_control

int main(int argc, char** argv) {
    // Initialize ROS2
    rclcpp::init(argc, argv);

    // Create and spin the node
    rclcpp::spin(std::make_shared<vtem_control::ValveControlNode>());

    // Shutdown ROS2
    rclcpp::shutdown();
    return 0;
}


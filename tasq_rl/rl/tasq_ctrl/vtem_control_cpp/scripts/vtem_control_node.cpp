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

        // Initialize ROS2 subscription for each valve pressure
        for (int i = 0; i < 16; ++i) {
            std::string topic_name = "valve_" + std::to_string(i) + "_pressure";
            valve_subscribers_[i] = this->create_subscription<std_msgs::msg::Int32>(
                topic_name, rclcpp::QoS(10), [this, i](std::shared_ptr<std_msgs::msg::Int32> msg) {
                    this->onPressureReceived(msg, i);
                });
        }

        // Periodic logging thread
        logThread_ = std::thread([this]() {
            while (rclcpp::ok()) {
                {
                    std::lock_guard<std::mutex> lock(dataMutex_);
                    for (const auto& [valveIdx, pressure] : valvePressureMap_) {
                        int actualPressure = vtemControl_.get_single_pressure(valveIdx);
                        RCLCPP_INFO(this->get_logger(),
                                    "Valve %d | Commanded Pressure: %d | Actual Pressure: %d mBar",
                                    valveIdx, pressure, actualPressure);
                    }
                }
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        });
    }

    ~ValveControlNode() {
        {
            std::lock_guard<std::mutex> lock(dataMutex_);
            for (const auto& [valveIdx, _] : valvePressureMap_) {
                unsigned int slotIdx = vtemControl_.get_slot_idx_from_valve_idx(valveIdx);
                vtemControl_.deactivate_pressure_regulation(slotIdx);
            }
        }
        vtemControl_.disconnect();

        if (logThread_.joinable()) {
            logThread_.join();
        }
    }

private:
    void onPressureReceived(const std_msgs::msg::Int32::SharedPtr msg, int valveIdx) {
        int value = msg->data;

        if (valveIdx < 12) {
            // Pressure regulation for valves 0-11
            unsigned int maxAllowedPressure = 400;
            if (value > maxAllowedPressure) {
                RCLCPP_WARN(this->get_logger(),
                            "Valve %d: Pressure %d mBar exceeds safety limit (%d mBar). Ignoring.",
                            valveIdx, value, maxAllowedPressure);
                return;
            }
            if (value < 0) {
                RCLCPP_WARN(this->get_logger(), "Pressure must be non-negative. Ignoring.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Received pressure for valve: %d", valveIdx);

            std::lock_guard<std::mutex> lock(dataMutex_);
            if (valvePressureMap_.find(valveIdx) == valvePressureMap_.end()) {
                RCLCPP_INFO(this->get_logger(), "Valve %d not active, activating pressure regulation", valveIdx);
                unsigned int slotIdx = vtemControl_.get_slot_idx_from_valve_idx(valveIdx);
                if (!vtemControl_.activate_pressure_regulation(slotIdx)) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to activate pressure regulation for valve %d.", valveIdx);
                    return;
                }
            }
            vtemControl_.set_single_pressure(valveIdx, value);
            valvePressureMap_[valveIdx] = value;
        } else {
            // Opening regulation for valves 12-15
            int opening_percent = std::max(-100, std::min(100, value));
            RCLCPP_INFO(this->get_logger(), "Received opening for valve: %d -> %d%%", valveIdx, opening_percent);

            std::lock_guard<std::mutex> lock(dataMutex_);
            if (valvePressureMap_.find(valveIdx) == valvePressureMap_.end()) {
                RCLCPP_INFO(this->get_logger(), "Valve %d not active, activating opening regulation", valveIdx);
                unsigned int slotIdx = vtemControl_.get_slot_idx_from_valve_idx(valveIdx);
                if (!vtemControl_.activate_valve_opening_regulation(slotIdx)) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to activate opening regulation for valve %d.", valveIdx);
                    return;
                }
            }
            vtemControl_.set_single_valve_opening(valveIdx, opening_percent);
            valvePressureMap_[valveIdx] = opening_percent;
        }
    }

    vtem_control::VtemControl vtemControl_;
    std::unordered_map<int, rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr> valve_subscribers_;
    std::unordered_map<unsigned int, unsigned int> valvePressureMap_;
    std::mutex dataMutex_;
    std::thread logThread_;
};

} // namespace vtem_control

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<vtem_control::ValveControlNode>());
    rclcpp::shutdown();
    return 0;
}

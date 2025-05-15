#ifndef VALVECONTROLWINDOW_HPP
#define VALVECONTROLWINDOW_HPP

#include <QWidget>
#include <QSlider>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <map>
#include <vector>

namespace vtem_control_gui {

class ValveControlWindow : public QWidget {
    Q_OBJECT

public:
    ValveControlWindow(rclcpp::Node::SharedPtr node, QWidget* parent = nullptr);
    ~ValveControlWindow();

private slots:
    void onSliderValueChanged(int value);
    //void onApplyAllPressed();

private:
    rclcpp::Node::SharedPtr node_;
    std::map<int, rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> valve_publishers_;
    QVBoxLayout* layout_;
    //QPushButton* applyButton_;
    std::vector<QSlider*> pressureSliders_;    // Vector of sliders for valve pressures
    std::vector<QLabel*> pressureValues_;      // Vector of labels for pressure values
};

} // namespace vtem_control_gui

#endif // VALVECONTROLWINDOW_HPP


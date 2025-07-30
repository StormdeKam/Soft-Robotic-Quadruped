#include "GaitLoaderWindow.hpp"
#include <QSlider>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QFileDialog>
#include <QComboBox>
#include <QTimer>
#include <fstream>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>

namespace gait_loader_gui {

GaitLoaderWindow::GaitLoaderWindow(rclcpp::Node::SharedPtr node, QWidget* parent)
    : QWidget(parent), node_(node), timer_(new QTimer(this)), currentStep_(0), playbackSpeed_(1.0) {
    
    const int numValves = 16;
    pressureSliders_.resize(numValves);
    pressureValues_.resize(numValves);
    toggleSwitches_.resize(numValves);

    layout_ = new QVBoxLayout();

    for (int i = 0; i < numValves; ++i) {
        valve_publishers_[i] = node->create_publisher<std_msgs::msg::Int32>(
            "valve_" + std::to_string(i) + "_pressure", rclcpp::QoS(10));

        QHBoxLayout* valveLayout = new QHBoxLayout();
        QLabel* label = new QLabel("Valve " + QString::number(i));
        valveLayout->addWidget(label);

        pressureSliders_[i] = new QSlider(Qt::Horizontal);
        int maxPressure = (i < 12) ? 400 : 3000;
        pressureSliders_[i]->setRange(0, maxPressure);
        connect(pressureSliders_[i], &QSlider::valueChanged, this, &GaitLoaderWindow::onSliderValueChanged);
        valveLayout->addWidget(pressureSliders_[i]);

        pressureValues_[i] = new QLabel("0 mBar");
        valveLayout->addWidget(pressureValues_[i]);

        toggleSwitches_[i] = new QCheckBox();
        valveLayout->addWidget(toggleSwitches_[i]);

        layout_->addLayout(valveLayout);
    }

    QPushButton* loadCsvButton = new QPushButton("Load CSV");
    connect(loadCsvButton, &QPushButton::clicked, this, &GaitLoaderWindow::loadCsvFile);
    layout_->addWidget(loadCsvButton);

    QPushButton* startPlaybackButton = new QPushButton("Start Playback");
    connect(startPlaybackButton, &QPushButton::clicked, this, &GaitLoaderWindow::startPlayback);
    layout_->addWidget(startPlaybackButton);

    playbackSpeedBox_ = new QComboBox();
    playbackSpeedBox_->addItem("0.5x", 200);
    playbackSpeedBox_->addItem("1x", 100);
    playbackSpeedBox_->addItem("2x", 50);
    playbackSpeedBox_->addItem("5x", 20);
    connect(playbackSpeedBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &GaitLoaderWindow::updatePlaybackSpeed);

    layout_->addWidget(playbackSpeedBox_);

    setLayout(layout_);
}

GaitLoaderWindow::~GaitLoaderWindow() {}

void GaitLoaderWindow::onSliderValueChanged(int value) {
    for (int i = 0; i < 16; ++i) {
        if (sender() == pressureSliders_[i]) {
            pressureValues_[i]->setText(QString::number(value) + " mBar");

            if (toggleSwitches_[i]->isChecked()) {
                std_msgs::msg::Int32 msg;
                msg.data = value;
                valve_publishers_[i]->publish(msg);

                RCLCPP_INFO(rclcpp::get_logger("GaitLoaderWindow"),
                            "Published value %d to valve %d", value, i);
            }
            return;
        }
    }
}    
    
void GaitLoaderWindow::loadCsvFile() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open CSV File", "", "CSV Files (*.csv)");
    if (fileName.isEmpty()) return;

    std::ifstream file(fileName.toStdString());
    if (!file.is_open()) {
        RCLCPP_ERROR(rclcpp::get_logger("GaitLoaderWindow"), "Failed to open CSV file");
        return;
    }

    valveData_.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<int> row;
        int value;
        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }
        valveData_.push_back(row);
    }
    file.close();

    RCLCPP_INFO(rclcpp::get_logger("GaitLoaderWindow"), "Loaded CSV with %zu time steps", valveData_.size());
}

void GaitLoaderWindow::startPlayback() {
    playbackSpeed_ = 200;
    if (valveData_.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("GaitLoaderWindow"), "No data loaded");
        return;
    }

    currentStep_ = 0;
    connect(timer_, &QTimer::timeout, this, &GaitLoaderWindow::publishNextStep);
    timer_->start(playbackSpeed_);
}

void GaitLoaderWindow::publishNextStep() {
    if (currentStep_ >= valveData_.size()) {
        timer_->stop();

        for (auto& slider : pressureSliders_) {
            slider->setEnabled(true);
        }
        return;
    }

    for (int i = 0; i < 16; ++i) {
        int value = valveData_[currentStep_][i];
        pressureSliders_[i]->setValue(value);
        pressureValues_[i]->setText(QString::number(value) + " mBar");

        std_msgs::msg::Int32 msg;
        msg.data = value;
        valve_publishers_[i]->publish(msg);

        RCLCPP_INFO(rclcpp::get_logger("GaitLoaderWindow"), "Published %d to valve %d", value, i);
    }

    currentStep_++;
}

void GaitLoaderWindow::updatePlaybackSpeed(int index) {
    std::cout << "Selected playback speed index: " << index << std::endl;
    playbackSpeed_ = playbackSpeedBox_->itemData(index).toInt();
    if (timer_->isActive()) {
        timer_->start(playbackSpeed_);
    }
}

} // namespace gait_loader_gui

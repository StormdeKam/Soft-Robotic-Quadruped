#include "VtemControl.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>

/*
   Runtime configurable test program for actuating valves.
   Allows changing valve indices and pressure dynamically during execution.
 */

// Function to parse indices from a comma-separated string
std::vector<unsigned int> parse_indices(const std::string& indices_str) {
    std::vector<unsigned int> indices;
    std::stringstream ss(indices_str);
    std::string item;

    while (std::getline(ss, item, ',')) {
        try {
            indices.push_back(std::stoul(item));
        } catch (const std::exception& e) {
            std::cerr << "Invalid index: " << item << std::endl;
        }
    }

    return indices;
}

int main() {
    // Create VtemControl controller
    vtem_control::VtemControl vtemControl("192.168.4.3", "502");

    // Connect to VTEM
    if (!vtemControl.connect()) {
        std::cerr << "Failed to connect to VTEM." << std::endl;
        return -1;
    }

    // Acknowledge errors for all slots
    vtemControl.acknowledge_errors();

    // Shared variables for dynamic updates
    std::unordered_map<unsigned int, unsigned int> valvePressureMap; // Maps valve indices to pressures
    std::mutex dataMutex;

    // Start periodic logging in a separate thread
    std::thread logThread([&]() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(dataMutex);
                for (const auto& [valveIdx, pressure] : valvePressureMap) {
                    int actualPressure = vtemControl.get_single_pressure(valveIdx);
                    std::cout << "Valve " << valveIdx
                              << " | Commanded Pressure: " << pressure
                              << " | Actual Pressure: " << actualPressure << " mBar" << std::endl;
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    });

    // Main control loop
    while (true) {
        std::cout << "Enter commanded pressure and valve indices (e.g., 100 0,1,2 or -1 to quit): ";
        std::string input;
        std::getline(std::cin, input);

        // Exit condition
        if (input == "-1") {
            std::cout << "Exiting program." << std::endl;
            break;
        }

        // Parse input
        std::istringstream iss(input);
        std::string pressureStr, indicesStr;
        if (!(iss >> pressureStr >> indicesStr)) {
            std::cerr << "Invalid input. Format: <pressure> <comma-separated indices>" << std::endl;
            continue;
        }

        // Validate and update pressure
        unsigned int newPressure;
        try {
            newPressure = std::stoul(pressureStr);
            if (newPressure > 450) {
                std::cerr << "Pressure must be between 0 and 450 mBar." << std::endl;
                continue;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid pressure value: " << pressureStr << std::endl;
            continue;
        }

        // Parse and update valve indices
        std::vector<unsigned int> newValveIndices = parse_indices(indicesStr);
        if (newValveIndices.empty()) {
            std::cerr << "No valid valve indices provided." << std::endl;
            continue;
        }

        // Apply new settings
        {
            std::lock_guard<std::mutex> lock(dataMutex);

            for (unsigned int valveIdx : newValveIndices) {
                if (valvePressureMap.find(valveIdx) == valvePressureMap.end()) {
                    // If the valve is not active, activate pressure regulation
                    unsigned int slotIdx = vtemControl.get_slot_idx_from_valve_idx(valveIdx);
                    if (!vtemControl.activate_pressure_regulation(slotIdx)) {
                        std::cerr << "Failed to activate pressure regulation for valve " << valveIdx << "." << std::endl;
                        continue;
                    }
                }

                // Update or set pressure for the valve
                vtemControl.set_single_pressure(valveIdx, newPressure);
                valvePressureMap[valveIdx] = newPressure;
            }
        }
    }

    // Cleanup
    {
        std::lock_guard<std::mutex> lock(dataMutex);
        for (const auto& [valveIdx, _] : valvePressureMap) {
            unsigned int slotIdx = vtemControl.get_slot_idx_from_valve_idx(valveIdx);
            vtemControl.deactivate_pressure_regulation(slotIdx);
        }
    }
    vtemControl.disconnect();

    // Stop logging thread
    logThread.detach();

    return 0;
} 

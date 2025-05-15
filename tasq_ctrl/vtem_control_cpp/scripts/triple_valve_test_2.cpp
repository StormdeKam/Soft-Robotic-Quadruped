#include "VtemControl.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*
   Command-line controlled program for actuating valves.
   Example usage:
       ./vtem_control 0,1,2 100
   - First argument: comma-separated valve indices (e.g., "0,1,2")
   - Second argument: commanded pressure in mBar (e.g., "100")
*/

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <valve_indices> <pressure>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 0,1,2 100" << std::endl;
        return -1;
    }

    // Parse valve indices
    std::vector<unsigned int> valveIndices;
    std::istringstream indicesStream(argv[1]);
    std::string token;
    while (std::getline(indicesStream, token, ',')) {
        try {
            unsigned int valveIdx = std::stoul(token);
            valveIndices.push_back(valveIdx);
        } catch (...) {
            std::cerr << "Invalid valve index: " << token << std::endl;
            return -1;
        }
    }

    // Parse commanded pressure
    unsigned int commandedPressure;
    try {
        commandedPressure = std::stoul(argv[2]);
        if (commandedPressure > 450) {
            std::cerr << "Pressure must be between 0 and 450 mBar." << std::endl;
            return -1;
        }
    } catch (...) {
        std::cerr << "Invalid pressure value: " << argv[2] << std::endl;
        return -1;
    }

    // Create VtemControl controller
    vtem_control::VtemControl vtemControl("192.168.4.3", "502");

    // Connect to the VTEM system
    if (!vtemControl.connect()) {
        std::cerr << "Failed to connect to VTEM." << std::endl;
        return -1;
    }

    // Acknowledge errors for all slots
    vtemControl.acknowledge_errors();

    // Activate pressure regulation for specified valves
    for (unsigned int valveIdx : valveIndices) {
        unsigned int slotIdx = vtemControl.get_slot_idx_from_valve_idx(valveIdx);
        if (!vtemControl.activate_pressure_regulation(slotIdx)) {
            std::cerr << "Failed to activate pressure regulation for valve " << valveIdx << "." << std::endl;
            vtemControl.disconnect();
            return -1;
        }
    }

    // Set commanded pressure for specified valves
    for (unsigned int valveIdx : valveIndices) {
        try {
            vtemControl.set_single_pressure(valveIdx, commandedPressure);
            std::cout << "Set pressure for valve " << valveIdx << " to " << commandedPressure << " mBar." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error setting pressure for valve " << valveIdx << ": " << e.what() << std::endl;
            vtemControl.disconnect();
            return -1;
        }
    }

    // Print actual pressure readings
    std::cout << "Reading actual pressures..." << std::endl;
    for (unsigned int valveIdx : valveIndices) {
        try {
            int actualPressure = vtemControl.get_single_pressure(valveIdx);
            std::cout << "Valve " << valveIdx << " Actual Pressure: " << actualPressure << " mBar" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error reading pressure for valve " << valveIdx << ": " << e.what() << std::endl;
        }
    }

    // Cleanup and disconnect
    for (unsigned int valveIdx : valveIndices) {
        unsigned int slotIdx = vtemControl.get_slot_idx_from_valve_idx(valveIdx);
        vtemControl.deactivate_pressure_regulation(slotIdx);
    }
    vtemControl.disconnect();

    return 0;
}


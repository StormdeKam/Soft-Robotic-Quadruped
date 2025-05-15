#include "VtemControl.hpp"
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include <sstream>
#include <vector>
#include <string>
#include <limits>

/*
   Test program for actuating twelve valves, skipping indices 6 and 7.
 */

int main() {
    // Create VtemControl controller.
    vtem_control::VtemControl vtemControl("192.168.4.3", "502");

    unsigned int valveIndices[] = {0, 1, 2, 3}; // Indices of valves to test, skipping 6 and 7

    unsigned int commandedPressure = 100; // tested 0 to 450 mbar

    // Connect.
    if (!vtemControl.connect()) {
        std::cout << "Failed to connect to VTEM." << std::endl;
        return -1;
    }

    // Acknowledge errors for all slots
    vtemControl.acknowledge_errors();

    // Activate pressure regulation for all valves
    for (unsigned int valveIdx : valveIndices) {
        unsigned int slotIdx = vtemControl.get_slot_idx_from_valve_idx(valveIdx);
        if (!vtemControl.activate_pressure_regulation(slotIdx)) {
            std::cout << "Failed to activate pressure regulation for valve " << valveIdx << "." << std::endl;
            return -1;
        }
    }

    int actualPressure;

    // Start periodic logging in a separate thread
    std::thread logThread([&]() {
        while (true) {
            for (unsigned int valveIdx : valveIndices) {
                actualPressure = vtemControl.get_single_pressure(valveIdx);
                std::cout << "Valve " << valveIdx
                          << " Actual Pressure: " << actualPressure << " mBar" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    });

    while (true) {
        std::cout << "Enter commanded pressure (0-450 mBar, or -1 to quit): ";
        std::cin >> commandedPressure;

        // Check if the user wants to exit
        if (commandedPressure == static_cast<unsigned int>(-1)) {
            std::cout << "Exiting test." << std::endl;
            break;
        }

        // Validate commanded pressure
        if (commandedPressure > 450) {
            std::cout << "Invalid pressure. Please enter a value between 0 and 450 mBar." << std::endl;
            continue;
        }

        for (unsigned int valveIdx : valveIndices) {
            // Set commanded pressure to the valve
            vtemControl.set_single_pressure(valveIdx, commandedPressure);
        }
    }

    // Cleanup
    for (unsigned int valveIdx : valveIndices) {
        unsigned int slotIdx = vtemControl.get_slot_idx_from_valve_idx(valveIdx);
        vtemControl.deactivate_pressure_regulation(slotIdx);
    }
    vtemControl.disconnect();

    // Stop the logging thread
    logThread.detach(); // Allow it to exit gracefully

    return 0;
}   


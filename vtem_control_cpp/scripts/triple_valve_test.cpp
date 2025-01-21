#include "VtemControl.hpp"
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>

#define STEP_TEST true // send step signals to valve

/*
   Test program for actuating three valves (indices 0, 1, and 2).
 */

int step_func(int i) {
    // step function to send to valve
    if (i % 1500 < 750) {
        return 300;
    } else {
        return 400;
    }
}

int main() {
    // Create VtemControl controller.
    vtem_control::VtemControl vtemControl("192.168.4.3", "502");

    unsigned int valveIndices[] = {0, 1, 2}; // Indices of valves to test
    const int numValves = sizeof(valveIndices) / sizeof(valveIndices[0]);

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
    const int cycles = 1000;
    std::vector<double> x(cycles), actualPressures[numValves], commandedPressures[numValves];

    for (int i = 0; i < numValves; i++) {
        actualPressures[i].resize(cycles);
        commandedPressures[i].resize(cycles);
    }

    time_t tstart, tend;
    tstart = time(0);
    for (int i = 0; i < cycles; i++) {
        // Wait 10 ms.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        if (STEP_TEST) {
            commandedPressure = step_func(i);
        }

        for (int v = 0; v < numValves; v++) {
            unsigned int valveIdx = valveIndices[v];

            // Log commanded pressure
            commandedPressures[v].at(i) = commandedPressure;

            // Read pressure of the valve
            actualPressure = vtemControl.get_single_pressure(valveIdx);
            x.at(i) = i;
            actualPressures[v].at(i) = actualPressure;

            // Set commanded pressure to the valve
            vtemControl.set_single_pressure(valveIdx, commandedPressure);

            if (i % 50 == 0) {
                std::cout << "Valve " << valveIdx << " actual pressure: " << actualPressure
                          << " mBar\tcommanded pressure: " << commandedPressure << " mBar" << std::endl;
            }
        }
    }

    tend = time(0);
    std::cout << "Test completed in " << difftime(tend, tstart) << " seconds." << std::endl;
    return 0;
}


import numpy as np
import csv
from rl.gaits.demo_gait_straight import demo_pressures, demo_suction

# Insert your full demo_pressures and demo_suction functions here

# --- Parameters ---
mode = 0  # direction of gait: 0 = forward
n = 10
gait_length = n * 23
output_file = "rl/gaits/gait_pattern.csv"

# Initialize absolute pressure state
pressures_abs = np.zeros((4, 3))

# Create CSV
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for step in range(gait_length):
        # Get pressure increments
        d_pressures = demo_pressures(step, mode)

        # Update absolute pressure
        pressures_abs += d_pressures

        # Optional: clip to safe range
        pressures_abs = np.clip(pressures_abs, 0, 400)

        # Get suction states
        suction_states = demo_suction(step, mode)
        suction_pressures = [500 if state else 0 for state in suction_states]

        # Combine absolute pressures + suction pressures into 16-element list
        flat_pressures = pressures_abs.flatten().tolist()
        full_row = flat_pressures + suction_pressures

        # Write row
        writer.writerow(full_row)

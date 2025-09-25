import numpy as np
import matplotlib.pyplot as plt
from rl.scripts.tasq_simulator import TASQSimulator  # adjust path if necessary
import pybullet as p
import time

# -------------------------
# PARAMETERS
# -------------------------
URDFS = {
    5: "rl/urdfs/TASQ_n5.urdf",
    6: "rl/urdfs/TASQ_n6.urdf",
    7: "rl/urdfs/TASQ_n7.urdf"
}

TEST_LEG = 0
NO_LINKS = 3
CONTROL_FREQ = 240
STEPS = 100
MAX_PRESSURE = 400
DT = 1.0 / CONTROL_FREQ
HOLD_ZERO_STEPS = 100
STABILIZE_STEPS = 200
NUM_RUNS = 10

# -------------------------
# HELPER FUNCTION
# -------------------------
def ramp_pressures(start, end, steps):
    return np.linspace(start, end, steps)

# -------------------------
# RUN TESTS
# -------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Hysteresis: Expansion/Contraction")
plt.xlabel("Pressure (mBar)")
plt.ylabel("Relative Z position (m)")
plt.grid(True)

plt.subplot(1,2,2)
plt.title("Hysteresis: Bending/Straightening")
plt.xlabel("Pressure in chamber 1 (mBar)")
plt.ylabel("Relative X position (m)")
plt.grid(True)

for urdf_num, urdf_path in URDFS.items():
    for run in range(NUM_RUNS):
        print(f"Running URDF {urdf_num}, run {run+1}/{NUM_RUNS}")
        sim = TASQSimulator(urdf_path, gui=False, no_links=NO_LINKS)

        sim.reset_simulation()
        p.resetBasePositionAndOrientation(sim.robot_id, [0,0,0.3], [0,0,0,1])
        p.createConstraint(sim.robot_id, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0.3])

        bottom_link_idx = TEST_LEG*(NO_LINKS*4 + 2) + NO_LINKS*4 + 1

        # Stabilize
        sim.apply_pressures(np.zeros((4,3)))
        sim.apply_suction(np.zeros(4))
        for _ in range(STABILIZE_STEPS):
            sim.step()

        # -------------------------
        # 1) Expansion/Contraction
        z_positions = []
        pressures_record = []

        pressure_ramp_up = ramp_pressures(0, MAX_PRESSURE, STEPS)
        for p_val in pressure_ramp_up:
            pressures = np.zeros((4,3))
            pressures[TEST_LEG,:] = p_val
            sim.apply_pressures(pressures)
            sim.step()
            pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
            z_positions.append(pos[2])
            pressures_record.append(p_val)

        pressure_ramp_down = ramp_pressures(MAX_PRESSURE, 0, STEPS)
        for p_val in pressure_ramp_down:
            pressures = np.zeros((4,3))
            pressures[TEST_LEG,:] = p_val
            sim.apply_pressures(pressures)
            sim.step()
            pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
            z_positions.append(pos[2])
            pressures_record.append(p_val)

        for _ in range(HOLD_ZERO_STEPS):
            pressures = np.zeros((4,3))
            sim.apply_pressures(pressures)
            sim.step()
            pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
            z_positions.append(pos[2])
            pressures_record.append(0)

        # Make relative to initial Z
        z_positions = np.array(z_positions) - z_positions[0]
        plt.subplot(1,2,1)
        plt.plot(pressures_record, z_positions, alpha=0.7)

        # -------------------------
        # 2) Bending/Straightening
        x_positions = []
        pressures_record = []

        pressure_ramp_up = ramp_pressures(0, MAX_PRESSURE, STEPS)
        for p_val in pressure_ramp_up:
            pressures = np.zeros((4,3))
            pressures[TEST_LEG,0] = p_val
            sim.apply_pressures(pressures)
            sim.step()
            pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
            x_positions.append(pos[0])
            pressures_record.append(p_val)

        pressure_ramp_down = ramp_pressures(MAX_PRESSURE, 0, STEPS)
        for p_val in pressure_ramp_down:
            pressures = np.zeros((4,3))
            pressures[TEST_LEG,0] = p_val
            sim.apply_pressures(pressures)
            sim.step()
            pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
            x_positions.append(pos[0])
            pressures_record.append(p_val)

        for _ in range(HOLD_ZERO_STEPS):
            pressures = np.zeros((4,3))
            sim.apply_pressures(pressures)
            sim.step()
            pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
            x_positions.append(pos[0])
            pressures_record.append(0)

        # Make relative to initial X
        x_positions = np.array(x_positions) - x_positions[0]
        plt.subplot(1,2,2)
        plt.plot(pressures_record, x_positions, alpha=0.7)

        sim.disconnect()

plt.tight_layout()
plt.show()

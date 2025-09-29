import numpy as np
import matplotlib.pyplot as plt
from rl.scripts.tasq_simulator import TASQSimulator  # adjust path if necessary
import pybullet as p

# -------------------------
# PARAMETERS
# -------------------------
URDF_PATH = "rl/urdfs/TASQ_n7.urdf"

CSV_PATH = "rl/datasets/actuator_validation/hysteresis_data.csv"

TEST_LEG = 0
NO_LINKS = 3
CONTROL_FREQ = 240
STEPS = 10
MAX_PRESSURE = 400
DT = 1.0 / CONTROL_FREQ
HOLD_ZERO_STEPS = 100
STABILIZE_STEPS = 200
NUM_RUNS = 1

# -------------------------
# HELPER FUNCTION
# -------------------------
def ramp_pressures(start, end, steps):
    return np.linspace(start, end, steps)

# -------------------------
# PLOT SETUP
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

# -------------------------
# SIMULATION RUNS
# -------------------------
for run in range(NUM_RUNS):
    sim = TASQSimulator(URDF_PATH, gui=True, no_links=NO_LINKS)

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

    pressure_ramp_down = ramp_pressures(MAX_PRESSURE-40, 0, STEPS-1)
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

    pressure_ramp_down = ramp_pressures(MAX_PRESSURE-40, 0, STEPS-1)
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

# -------------------------
# REAL DATA (CSV)
# -------------------------
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)

# Expansion test: 
z_0 = -data[0, 5]
z_expansion = -data[:75, 5]-z_0   # col 6 , multiply by -1
p_expansion = data[:75, 6]   # col 7

# Bending test:
x_0 = data[0,8]
x_deflection = data[:94, 8]-x_0  # col 9
p_bending = data[:94, 9]      # col 10

plt.subplot(1,2,1)
plt.plot(p_expansion, z_expansion, 'k--', linewidth=2, label="Real data")

plt.subplot(1,2,2)
plt.plot(p_bending, x_deflection, 'k--', linewidth=2, label="Real data")

# -------------------------
# FINALIZE
# -------------------------
plt.subplot(1,2,1)
plt.legend()

plt.subplot(1,2,2)
plt.legend()

plt.tight_layout()
plt.show()

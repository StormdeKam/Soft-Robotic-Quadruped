import numpy as np
import matplotlib.pyplot as plt
from rl.scripts.tasq_simulator import TASQSimulator  # adjust path if necessary
import pybullet as p

# -------------------------
# PARAMETERS
# -------------------------
URDF_PATH = "rl/urdfs/TASQ_n5.urdf"
CSV_PATH = "rl/datasets/actuator_validation/hysteresis_data.csv"

TEST_LEG = 0
NO_LINKS = 3
CONTROL_FREQ = 240
SIM_STEPS = 500      # number of simulation steps to record oscillations
STABILIZE_STEPS = 200  # stabilize before starting
DT = 1.0 / CONTROL_FREQ

# Max initial values
PRISMATIC_MAX = 0.005  # meters (as in apply_suction)
REVOLUTE_MAX_Y = 0.5   # rad (example maximum angle for y-axis joints)

# -------------------------
# INITIALIZE SIMULATOR
# -------------------------
sim = TASQSimulator(URDF_PATH, gui=True, no_links=NO_LINKS)
sim.reset_simulation()

# Fix the base in the air
p.resetBasePositionAndOrientation(sim.robot_id, [0,0,0.3], [0,0,0,1])
p.createConstraint(sim.robot_id, -1, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0.3])

bottom_link_idx = TEST_LEG*(NO_LINKS*4 + 2) + NO_LINKS*4 + 1

# -------------------------
# STABILIZE LEGS
# -------------------------
sim.apply_pressures(np.zeros((4,3)))
sim.apply_suction(np.zeros(4))
for _ in range(STABILIZE_STEPS):
    sim.step()

# -------------------------
# 1) PRISMATIC OSCILLATION (z position)
# -------------------------
prismatic_per_leg = len(sim.prismatic_joints) // 4
for j in range(prismatic_per_leg):
    joint_idx = sim.prismatic_joints[TEST_LEG*prismatic_per_leg + j]
    p.resetJointState(sim.robot_id, joint_idx, targetValue=PRISMATIC_MAX/3)

z_positions = []
for step in range(SIM_STEPS):
    sim.apply_pressures(np.zeros((4,3)))
    sim.step()
    pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
    z_positions.append(pos[2])

z_positions = np.array(z_positions)
# Make relative to initial simulation position
z_positions_rel = z_positions - z_positions[-1]
time_sim = np.arange(SIM_STEPS) * DT

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(time_sim, z_positions_rel, label="Simulated")
plt.axhline(0, color='k', linestyle='--', label="Zero reference")
plt.xlabel("Time (s)")
plt.ylabel("Z position (m)")
plt.title("Prismatic oscillation")
plt.grid(True)

# -------------------------
# 2) REVOLUTE Y OSCILLATION (x position)
# -------------------------
# Reset joints
sim.apply_pressures(np.zeros((4,3)))
sim.apply_suction(np.zeros(4))
for joint_idx in sim.revolute_joints:
    if joint_idx % 2 == 1:  # y-axis joints
        p.resetJointState(sim.robot_id, joint_idx, targetValue=REVOLUTE_MAX_Y)
    else:
        p.resetJointState(sim.robot_id, joint_idx, targetValue=0.0)

x_positions = []
for step in range(SIM_STEPS):
    sim.apply_pressures(np.zeros((4,3)))
    sim.step()
    pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
    x_positions.append(pos[0])

x_positions = np.array(x_positions)
# Make relative to initial simulation position
x_positions_rel = x_positions - x_positions[-1]

plt.subplot(1,2,2)
plt.plot(time_sim, x_positions_rel, label="Simulated")
plt.axhline(0, color='k', linestyle='--', label="Zero reference")
plt.xlabel("Time (s)")
plt.ylabel("X position (m)")
plt.title("Revolute Y oscillation")
plt.grid(True)

# -------------------------
# REAL DATA (CSV)
# -------------------------
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)

# Columns for oscillation:
# Expansion test: 2(time), 3(deflection z)
# Bending test: 0(time), 1(deflection x)

# Expansion (z)
time_real_z = data[:114,2]
z_deflection_real = data[:114,3]
plt.subplot(1,2,1)
plt.plot(time_real_z, z_deflection_real, 'k--', linewidth=2, label="Real data")

# Bending (x)
time_real_x = data[:,0]
x_deflection_real = -data[:,1]
plt.subplot(1,2,2)
plt.plot(time_real_x, x_deflection_real, 'k--', linewidth=2, label="Real data")

# -------------------------
# FINALIZE
# -------------------------
plt.subplot(1,2,1)
plt.legend()
plt.subplot(1,2,2)
plt.legend()

plt.tight_layout()
plt.show()

sim.disconnect()

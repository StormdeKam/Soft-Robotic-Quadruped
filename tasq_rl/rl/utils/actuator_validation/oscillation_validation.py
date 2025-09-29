import numpy as np
import matplotlib.pyplot as plt
from rl.scripts.tasq_simulator import TASQSimulator  # adjust path if necessary
import pybullet as p
import time

# -------------------------
# PARAMETERS
# -------------------------
URDF_PATH = "rl/urdfs/TASQ_n7.urdf"
TEST_LEG = 0
NO_LINKS = 3
CONTROL_FREQ = 240
DT = 1.0 / CONTROL_FREQ
SIM_STEPS = 500      # number of simulation steps to record oscillations
STABILIZE_STEPS = 200  # stabilize before starting

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
# Set prismatic joints to max expansion
prismatic_per_leg = len(sim.prismatic_joints) // 4
for j in range(prismatic_per_leg):
    joint_idx = sim.prismatic_joints[TEST_LEG*prismatic_per_leg + j]
    p.resetJointState(sim.robot_id, joint_idx, targetValue=PRISMATIC_MAX/2)

# Record z position over time
z_positions = []
for _ in range(SIM_STEPS):
    sim.apply_pressures(np.zeros((4,3)))
    sim.step()
    pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
    z_positions.append(pos[2])

z_positions = np.array(z_positions)
plt.figure()
plt.plot(np.arange(SIM_STEPS)*DT, z_positions - z_positions[-1], label="Prismatic start max")
plt.axhline(0, color='k', linestyle='--', label="Zero position")
plt.xlabel("Time (s)")
plt.ylabel("Z position relative to final (m)")
plt.title("Oscillation from prismatic max expansion")
plt.grid(True)
plt.legend()

# -------------------------
# 2) REVOLUTE Y OSCILLATION (x position)
# -------------------------
# Reset to zero first
sim.apply_pressures(np.zeros((4,3)))
sim.apply_suction(np.zeros(4))
for joint_idx in sim.revolute_joints:
    joint_info = p.getJointInfo(sim.robot_id, joint_idx)
    joint_name = joint_info[1].decode()
    # Only y-axis joints (assuming even index corresponds to x, odd to y)
    if joint_idx % 2 == 1:
        p.resetJointState(sim.robot_id, joint_idx, targetValue=REVOLUTE_MAX_Y/2)
    else:
        p.resetJointState(sim.robot_id, joint_idx, targetValue=0.0)

x_positions = []
for _ in range(SIM_STEPS):
    sim.apply_pressures(np.zeros((4,3)))
    sim.step()
    pos, _ = p.getLinkState(sim.robot_id, bottom_link_idx)[:2]
    x_positions.append(pos[0])

x_positions = np.array(x_positions)
plt.figure()
plt.plot(np.arange(SIM_STEPS)*DT, x_positions - x_positions[-1], label="Revolute Y start max")
plt.axhline(0, color='k', linestyle='--', label="Zero position")
plt.xlabel("Time (s)")
plt.ylabel("X position relative to final (m)")
plt.title("Oscillation from revolute Y max angle")
plt.grid(True)
plt.legend()

plt.show()

sim.disconnect()

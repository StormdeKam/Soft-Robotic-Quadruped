import numpy as np
import matplotlib.pyplot as plt
import os

# --- Paths ---
dataset_folder = "rl/datasets/noise_control"
states_file = os.path.join(dataset_folder, "demo_states_low_freq.npy")

# --- Load dataset ---
states = np.load(states_file)
print("Loaded states:", states.shape)

# --- Extract data ---
base_pos = states[:, 0:3]           # base position (x, y, z)
imu_ori = states[:, 3:7]            # orientation (quaternion)
imu_ang_vel = states[:, 10:13]      # angular velocity
pressures = states[:, 13:25]        # 12 values (4x3)
suctions = states[:, 25:29]         # 4 values
grfs = states[:, 29:33]             # 4 values
phase = states[:, 33:35]            # (sin_phase, cos_phase)
goal_xy = states[:, 35:37]          # local goal (x, y)
goal_yaw = states[:, 37:39]         # (sin_delta_yaw, cos_delta_yaw)

# --- Time axis ---
time = np.arange(states.shape[0])

# --- Plot ---
fig, axs = plt.subplots(7, 1, figsize=(12, 26))

# 1. IMU Orientation
for i in range(imu_ori.shape[1]):
    axs[0].plot(time, imu_ori[:, i], label=f"Ori_{i}")
axs[0].set_title("IMU Orientation (Quaternion)")
axs[0].set_ylabel("Value")
axs[0].legend()

# 2. IMU Angular Velocity
for i in range(imu_ang_vel.shape[1]):
    axs[1].plot(time, imu_ang_vel[:, i], label=f"AngVel_{i}")
axs[1].set_title("IMU Angular Velocity")
axs[1].set_ylabel("rad/s")
axs[1].legend()

# 3. Ground Reaction Forces
for i in range(grfs.shape[1]):
    axs[2].plot(time, grfs[:, i], label=f"GRF_{i}")
axs[2].set_title("Ground Reaction Forces (Normalized)")
axs[2].set_ylabel("Force")
axs[2].legend()

# 4. Pressures
for i in range(pressures.shape[1]):
    axs[3].plot(time, pressures[:, i], label=f"P_{i}")
axs[3].set_title("Normalized Pressures (12 actuators)")
axs[3].set_ylabel("Pressure [0-1]")
axs[3].legend(ncol=4, fontsize=8)

# 5. Suctions
for i in range(suctions.shape[1]):
    axs[4].plot(time, suctions[:, i], label=f"Suction_{i}")
axs[4].set_title("Normalized Suction States")
axs[4].set_ylabel("State [0-1]")
axs[4].legend()

# 6. Gait Phase
axs[5].plot(time, phase[:, 0], label="sin(phase)")
axs[5].plot(time, phase[:, 1], label="cos(phase)")
axs[5].set_title("Gait Phase (sin, cos)")
axs[5].set_ylabel("Value [-1, 1]")
axs[5].legend()

# 7. Local Goal Information
axs[6].plot(time, goal_xy[:, 0], label="Goal X")
axs[6].plot(time, goal_xy[:, 1], label="Goal Y")
axs[6].plot(time, goal_yaw[:, 0], label="sin(Δyaw)")
axs[6].plot(time, goal_yaw[:, 1], label="cos(Δyaw)")
axs[6].set_title("Local Goal and Relative Orientation")
axs[6].set_xlabel("Timestep")
axs[6].set_ylabel("Value")
axs[6].legend()

plt.tight_layout()
plt.show()


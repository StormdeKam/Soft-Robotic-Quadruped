import numpy as np
import matplotlib.pyplot as plt
import os

# --- Paths ---
dataset_folder = "rl/datasets/noise_control"
states_file = os.path.join(dataset_folder, "demo_states_merged.npy")

# --- Load dataset ---
states = np.load(states_file)
print("Loaded states:", states.shape)

# --- Extract data ---
imu_ori = states[:, 3:7]         # orientation (quaternion)
imu_ang_vel = states[:, 10:13]   # angular velocity
pressures = states[:, 13:25]     # 12 values (4x3)
suctions = states[:, 25:29]      # 4 values
grfs = states[:, 29:33]          # 4 values

# --- Time axis ---
time = np.arange(states.shape[0])

# --- Plot ---
fig, axs = plt.subplots(5, 1, figsize=(12, 20))

# IMU Orientation
for i in range(imu_ori.shape[1]):
    axs[0].plot(time, imu_ori[:, i], label=f"Ori_{i}")
axs[0].set_title("IMU Orientation (Quaternion)")
axs[0].set_xlabel("Timestep")
axs[0].set_ylabel("Value")
axs[0].legend()

# IMU Angular Velocity
for i in range(imu_ang_vel.shape[1]):
    axs[1].plot(time, imu_ang_vel[:, i], label=f"AngVel_{i}")
axs[1].set_title("IMU Angular Velocity")
axs[1].set_xlabel("Timestep")
axs[1].set_ylabel("rad/s")
axs[1].legend()

# GRFs
for i in range(grfs.shape[1]):
    axs[2].plot(time, grfs[:, i], label=f"GRF_{i}")
axs[2].set_title("Ground Reaction Forces")
axs[2].set_xlabel("Timestep")
axs[2].set_ylabel("Normalized Force")
axs[2].legend()

# Pressures
for i in range(pressures.shape[1]):
    axs[3].plot(time, pressures[:, i], label=f"Pressure_{i}")
axs[3].set_title("Normalized Pressures")
axs[3].set_xlabel("Timestep")
axs[3].set_ylabel("Pressure [0-1]")
axs[3].legend()

# Suctions
for i in range(suctions.shape[1]):
    axs[4].plot(time, suctions[:, i], label=f"Suction_{i}")
axs[4].set_title("Normalized Suction States")
axs[4].set_xlabel("Timestep")
axs[4].set_ylabel("Suction [0-1]")
axs[4].legend()

plt.tight_layout()
plt.show()

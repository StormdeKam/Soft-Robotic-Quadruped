import numpy as np
import matplotlib.pyplot as plt
import os

# Path to saved dataset
dataset_dir = "rl/datasets/goal_directed"

# Load merged state data and episode lengths
states = np.load(os.path.join(dataset_dir, "demo_states_merged.npy"))
episode_lengths = np.load(os.path.join(dataset_dir, "demo_episode_lengths_merged.npy"))

print(f"Loaded states shape: {states.shape}")
print(f"Loaded episode lengths shape: {episode_lengths.shape}")
assert np.sum(episode_lengths) == states.shape[0], "Mismatch in episode lengths vs. state data"

def extract_yaw(sin_yaw, cos_yaw):
    return np.arctan2(sin_yaw, cos_yaw)

start_idx = 0
plt.figure(figsize=(10, 10))

for i, ep_len in enumerate(episode_lengths):
    end_idx = start_idx + ep_len
    ep_states = states[start_idx:end_idx]
    start_idx = end_idx

    # Extract base positions (x, y)
    x = ep_states[:, 0]
    y = ep_states[:, 1]

    # Get start pos and yaw
    x0, y0 = x[0], y[0]
    sin_yaw0, cos_yaw0 = ep_states[0, -2], ep_states[0, -1]
    yaw0 = extract_yaw(sin_yaw0, cos_yaw0)

    # Translate to origin
    dx = x - x0
    dy = y - y0

    # Rotate back by -yaw0
    cos_theta = np.cos(-yaw0)
    sin_theta = np.sin(-yaw0)
    dx_rot = cos_theta * dx - sin_theta * dy
    dy_rot = sin_theta * dx + cos_theta * dy

    # Optional: skip some episodes for clarity
    # if i % 10 != 0:
    #     continue

    plt.plot(dx_rot, dy_rot, alpha=0.4, linewidth=1)

plt.title("Normalized Base Trajectories (Δx, Δy) Aligned to Start Frame")
plt.xlabel("ΔX (m)")
plt.ylabel("ΔY (m)")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

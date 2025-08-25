import numpy as np
import matplotlib.pyplot as plt
import os

# Path to saved dataset
dataset_dir = "rl/datasets/minimal_viable"

# Load merged state data and episode lengths
states = np.load(os.path.join(dataset_dir, "demo_states_merged.npy"))
episode_lengths = np.load(os.path.join(dataset_dir, "demo_episode_lengths_merged.npy"))

# Sanity checks
print(f"Loaded states shape: {states.shape}")
print(f"Loaded episode lengths shape: {episode_lengths.shape}")
assert np.sum(episode_lengths) == states.shape[0], "Mismatch in total episode lengths vs. states"

# Plot each episode trajectory
start_idx = 0
plt.figure(figsize=(10, 10))
for i, ep_len in enumerate(episode_lengths):
    end_idx = start_idx + ep_len
    ep_states = states[start_idx:end_idx]
    x = ep_states[:, 0]  # base_pos_x
    y = ep_states[:, 1]  # base_pos_y

    plt.plot(x, y, alpha=0.4, linewidth=1, color='blue')

    start_idx = end_idx

plt.title("Base Trajectories (x, y) Across Episodes")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

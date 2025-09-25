import numpy as np
import matplotlib.pyplot as plt
import os

# Path to saved dataset
dataset_dir = "rl/datasets/minimal_viable_v7"

# Load merged state data and episode lengths
states = np.load(os.path.join(dataset_dir, "demo_states_merged.npy"))
episode_lengths = np.load(os.path.join(dataset_dir, "demo_episode_lengths_merged.npy"))

# Sanity checks
print(f"Loaded states shape: {states.shape}")
print(f"Loaded episode lengths shape: {episode_lengths.shape}")
assert np.sum(episode_lengths) == states.shape[0], "Mismatch in total episode lengths vs. states"

# Parameters
gaitcycle_len = 1400*5  # datapoints per gait cycle
cmap = plt.colormaps.get_cmap("viridis")  # smooth sequential colormap

plt.figure(figsize=(10, 10))
start_idx = 0

for i, ep_len in enumerate(episode_lengths):
    end_idx = start_idx + ep_len
    ep_states = states[start_idx:end_idx]
    x = ep_states[:, 0]  # base_pos_x
    y = ep_states[:, 1]  # base_pos_y

    # Number of cycles in this episode
    n_cycles = int(np.ceil(ep_len / gaitcycle_len))
    cycle_colors = cmap(np.linspace(0, 1, n_cycles))

    # Plot each gait cycle as one solid color line
    for c in range(n_cycles):
        cycle_start = c * gaitcycle_len
        cycle_end = min((c + 1) * gaitcycle_len, ep_len)

        if cycle_end - cycle_start < 2:
            continue

        cx = x[cycle_start:cycle_end]
        cy = y[cycle_start:cycle_end]

        plt.plot(cx, cy, color=cycle_colors[c], linewidth=1.5, alpha=0.9)

    start_idx = end_idx

plt.title("Base Trajectories (x, y) Across Episodes with Per-Cycle Coloring")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

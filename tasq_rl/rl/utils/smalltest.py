import numpy as np

episode_lengths = np.load("rl/datasets/goal_directed/demo_episode_lengths_merged.npy")
episode_start_indices = np.cumsum(np.insert(episode_lengths, 0, 0))[:-1]  # length = num episodes
print(episode_start_indices)
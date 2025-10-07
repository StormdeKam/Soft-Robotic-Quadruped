import numpy as np
import torch
from torch.utils.data import Dataset

class StackedDataset(Dataset):
    def __init__(self, state_path, action_path, episode_lengths_path, N=4, use_mmap=False, observed_indices=None):
        # load states and actions
        if use_mmap:
            self.states = np.load(state_path, mmap_mode='r')
            self.actions = np.load(action_path, mmap_mode='r')
        else:
            self.states = np.load(state_path)
            self.actions = np.load(action_path)

        self.N = N
        self.obs_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

        # Handle episode boundaries
        if episode_lengths_path is not None:
            self.episode_lengths = np.load(episode_lengths_path)
            self.episode_start_indices = np.cumsum(np.concatenate([[0], self.episode_lengths[:-1]]))
        else:
            self.episode_lengths = None
            self.episode_start_indices = []

        # Observed indices for a single timestep
        if observed_indices is None:
            self.single_step_indices = np.arange(self.obs_dim)
        else:
            self.single_step_indices = np.array(observed_indices, dtype=int)

        # Precompute stacked indices
        self.stacked_indices = np.concatenate([
            self.single_step_indices + i*self.obs_dim for i in range(N)
        ])

        # length: all states
        self.length = len(self.states)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # check if this idx is at episode start
        if any(idx < start + self.N-1 and idx >= start for start in self.episode_start_indices):
            start = next(start for start in self.episode_start_indices if idx < start + self.N-1 and idx >= start)
            num_missing = start + self.N - 1 - idx
            # fill missing states by repeating first available state
            pad_states = np.repeat(self.states[start:start+1], num_missing, axis=0)
            available_states = self.states[start:idx+1]
            stacked_obs = np.concatenate([pad_states, available_states], axis=0)
        else:
            stacked_obs = self.states[idx - self.N + 1 : idx + 1]

        stacked_obs = stacked_obs.reshape(-1).astype(np.float32)
        stacked_obs = stacked_obs[self.stacked_indices]
        action = self.actions[idx].astype(np.float32)
        return torch.from_numpy(stacked_obs), torch.from_numpy(action)


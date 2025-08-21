import numpy as np
import torch
from torch.utils.data import Dataset

class StackedDataset(Dataset):
    def __init__(self, state_path, action_path, N=4, use_mmap=False, observed_indices=None):
        if use_mmap:
            print("🔄 Loading with mmap_mode='r' (disk-backed, minimal RAM)")
            self.states = np.load(state_path, mmap_mode='r')
            self.actions = np.load(action_path, mmap_mode='r')
        else:
            print("🚀 Loading fully into RAM for faster access")
            self.states = np.load(state_path)
            self.actions = np.load(action_path)
        
        self.N = N
        self.obs_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        self.length = len(self.states) - N + 1

        # Observed indices for a single timestep
        if observed_indices is None:
            self.single_step_indices = np.arange(self.obs_dim)  # use all by default
        else:
            self.single_step_indices = np.array(observed_indices, dtype=int)

        # Precompute stacked indices for efficiency
        self.stacked_indices = np.concatenate([
            self.single_step_indices + i*self.obs_dim for i in range(N)
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx + self.N - 1
        stacked_obs = self.states[idx - self.N + 1 : idx + 1].reshape(-1).astype(np.float32)
        # select only observed state
        stacked_obs = stacked_obs[self.stacked_indices]
        action = self.actions[idx].astype(np.float32)
        return torch.from_numpy(stacked_obs), torch.from_numpy(action)

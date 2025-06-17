from torch.utils.data import Dataset
import numpy as np
import torch

class StackedDataset(Dataset):
    def __init__(self, state_path, action_path, N=4):
        self.states = np.load(state_path, mmap_mode='r')  # load lazily
        self.actions = np.load(action_path, mmap_mode='r')
        self.N = N
        self.length = len(self.states) - N + 1
        self.obs_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx + self.N - 1
        stacked_obs = self.states[idx - self.N + 1 : idx + 1].reshape(-1).astype(np.float32)
        
        # Add noise to the 4 time signals (assumed to be the last feature of each frame)
        obs_dim = self.obs_dim
        for i in range(self.N):
            time_index = (i + 1) * obs_dim - 1  # last feature in each frame
            stacked_obs[time_index] += np.random.normal(loc=0.0, scale=0.001)

        action = self.actions[idx].astype(np.float32)
        return torch.from_numpy(stacked_obs), torch.from_numpy(action)


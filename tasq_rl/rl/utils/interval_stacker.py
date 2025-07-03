import numpy as np
import torch
from torch.utils.data import Dataset

class IntervalStackedDataset(Dataset):
    def __init__(self, state_path, action_path, intervals=[40,80,120,160,200], use_mmap=False):
        if use_mmap:
            print("🔄 Loading with mmap_mode='r' (disk-backed, minimal RAM)")
            self.states = np.load(state_path, mmap_mode='r')
            self.actions = np.load(action_path, mmap_mode='r')
        else:
            print("🚀 Loading fully into RAM for faster access")
            self.states = np.load(state_path)
            self.actions = np.load(action_path)
        
        self.intervals = intervals
        self.max_offset = max(intervals)
        
        self.obs_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        
        # figure out indices of partial features
        self.partial_idx = np.r_[
            0:3,    # base pos
            3:7,    # base orn
            13:25,  # pressures (example!)
            25:29,   # suctions (example!)
            33:37   # goal_pos and orn
        ]
        
        self.length = len(self.states) - self.max_offset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # current full state
        current_state = self.states[idx + self.max_offset].astype(np.float32)
        prev_state = self.states[idx + self.max_offset -1].astype(np.float32)

        # partial histories
        partial_states = []
        for offset in self.intervals:
            past_idx = idx + self.max_offset - offset
            partial_state = self.states[past_idx][self.partial_idx]
            partial_states.append(partial_state.astype(np.float32))
        
        # concatenate
        stacked_obs = np.concatenate([current_state] + [prev_state] + partial_states)
        action = self.actions[idx + self.max_offset].astype(np.float32)

        return torch.from_numpy(stacked_obs), torch.from_numpy(action)

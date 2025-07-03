import numpy as np
import torch
from torch.utils.data import Dataset

class IntervalStackedDataset(Dataset):
    def __init__(self, state_path, action_path, episode_lengths_path, intervals=[40,80,120,160,200], use_mmap=False):
        if use_mmap:
            print("🔄 Loading with mmap_mode='r' (disk-backed, minimal RAM)")
            self.states = np.load(state_path, mmap_mode='r')
            self.actions = np.load(action_path, mmap_mode='r')
            self.episode_lengths = np.load(episode_lengths_path, mmap_mode='r')
        else:
            print("🚀 Loading fully into RAM for faster access")
            self.states = np.load(state_path)
            self.actions = np.load(action_path)
            self.episode_lengths = np.load(episode_lengths_path)
        
        self.intervals = intervals
        self.max_offset = max(intervals)
        
        self.obs_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        
        # Precompute cumulative episode start indices
        self.episode_start_indices = np.cumsum(np.insert(self.episode_lengths, 0, 0))[:-1]  # length = num episodes
        
        # figure out indices of partial features (adjust to your actual dataset)
        self.partial_idx = np.r_[
            0:3,    # base pos
            3:7,    # base orn
            13:25,  # pressures (example!)
            25:29,  # suctions (example!)
            33:37   # goal_pos and orn
        ]

        # Total length excludes last max_offset frames that don't have full history
        self.length = len(self.states) - self.max_offset

    def __len__(self):
        return self.length

    def _find_episode_start(self, idx):
        # Binary search for episode index containing idx
        # episode_start_indices are sorted start indices of each episode
        ep_idx = np.searchsorted(self.episode_start_indices, idx, side='right') - 1
        ep_start = self.episode_start_indices[ep_idx]
        ep_len = self.episode_lengths[ep_idx]
        return ep_start, ep_len

    def __getitem__(self, idx):
        real_idx = idx + self.max_offset  # offset for history window
        
        ep_start, ep_len = self._find_episode_start(real_idx)

        # Current full state
        current_state = self.states[real_idx].astype(np.float32)
        # Previous full state (one step before)
        prev_state = self.states[real_idx - 1].astype(np.float32)

        # Partial histories
        partial_states = []
        for offset in self.intervals:
            past_idx = real_idx - offset
            # If past_idx crosses episode start, clamp to episode start (steady state)
            if past_idx < ep_start:
                past_idx = ep_start
            partial_state = self.states[past_idx][self.partial_idx]
            partial_states.append(partial_state.astype(np.float32))
        
        # Concatenate full current state, previous state, and partial histories
        stacked_obs = np.concatenate([current_state] + [prev_state] + partial_states)
        action = self.actions[real_idx].astype(np.float32)

        return torch.from_numpy(stacked_obs), torch.from_numpy(action)

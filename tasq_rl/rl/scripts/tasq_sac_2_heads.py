import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from gym import spaces
import numpy as np


# --- Shared Feature Extractor (optional, or use default MlpExtractor) ---
class SharedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.shared(x)


# --- Custom Actor with Multi-Head Output ---
class MultiHeadActor(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.shared = SharedMLP(feature_dim)

        # Continuous pressure deltas (12-d)
        self.pressure_head = nn.Sequential(
            nn.Linear(256, 12),
            nn.Tanh()  # Assume range [-1, 1] for scaling to env
        )

        # Binary suction actions (4-d)
        self.suction_head = nn.Sequential(
            nn.Linear(256, 4),
            nn.Sigmoid()  # Probability output for binary decision
        )

    def forward(self, x):
        shared = self.shared(x)
        pressures = self.pressure_head(shared)
        suctions = self.suction_head(shared)
        return torch.cat([pressures, suctions], dim=-1)  # (batch, 16)
    
    def load_from_bc(self, bc_weights_path, device="cpu"):
        """Load weights from a pretrained BC model into the SAC actor."""
        state_dict = torch.load(bc_weights_path, map_location=device)

        own_state = self.state_dict()
        matched_keys = [k for k in own_state.keys() if k in state_dict and own_state[k].shape == state_dict[k].shape]
        for k in matched_keys:
            own_state[k] = state_dict[k]
        self.load_state_dict(own_state)
        print(f"[INFO] Loaded BC weights into MultiHeadActor from {bc_weights_path}")



# --- Custom Policy class to integrate with SB3 ---
class MultiHeadSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        input_dim = observation_space.shape[0]
        action_dim = 16

        self.actor = MultiHeadActor(input_dim, action_dim).to(self.device)
        self.actor_target = MultiHeadActor(input_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic setup (reuse SB3 MlpCritic)
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        # Actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))

    def _predict(self, observation, deterministic=False):
        with torch.no_grad():
            action = self.actor(observation)
        if deterministic:
            return action
        else:
            # Add Gaussian noise to pressure, sample suction
            pressures = action[:, :12] + 0.1 * torch.randn_like(action[:, :12])
            suctions = (action[:, 12:] > 0.5).float()
            return torch.cat([pressures, suctions], dim=-1)

    def forward(self, obs, deterministic=False):
        return self._predict(obs, deterministic)


# --- Custom SAC Class ---
from stable_baselines3.sac.sac import SAC as BaseSAC

class MultiHeadSAC(BaseSAC):
    policy_aliases = {
        "MultiHeadSACPolicy": MultiHeadSACPolicy
    }

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)


# --- Example usage ---
if __name__ == "__main__":
    from rl.scripts.tasq_env import TASQEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    def make_env():
        return TASQEnv(urdf_path="tasq_rl/urdfs/TASQ_n7.urdf", no_links=7, gui=False)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    model = MultiHeadSAC(
        policy=MultiHeadSACPolicy,
        env=env,
        verbose=1,
        buffer_size=100000,
        learning_rate=3e-4,
        batch_size=64,
        ent_coef="auto",
        train_freq=(1, "step"),
        gradient_steps=1
    )

    # Optional: Load pretrained BC weights
    bc_weights_path = "tasq_rl/models/tasq_bc/no_steady_states/tasq_bc_ep100_20250619-025729.pt"
    model.policy.actor.load_from_bc(bc_weights_path, device=model.device)
    model.policy.actor_target.load_state_dict(model.policy.actor.state_dict())

    model.learn(total_timesteps=10_000)
    model.save("tasq_rl/models/tasq_sac_multihead")

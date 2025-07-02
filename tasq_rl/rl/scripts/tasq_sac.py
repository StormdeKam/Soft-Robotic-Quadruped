import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import BasePolicy
from gym import spaces


# --- Shared Feature Extractor ---
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


# --- Multi-Head Actor ---
class MultiHeadActor(nn.Module):
    def __init__(self, feature_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.shared = SharedMLP(feature_dim)
        self.mu = nn.Linear(256, 12)
        self.log_std = nn.Linear(256, 12)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.suction_head = nn.Sequential(
            nn.Linear(256, 4),
            nn.Sigmoid()
        )

    def forward(self, obs, deterministic=False):
        x = self.shared(obs)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            pressure = torch.tanh(mu)
        else:
            normal = torch.distributions.Normal(mu, std)
            pressure_raw = normal.rsample()
            pressure = torch.tanh(pressure_raw)

        suction = self.suction_head(x)
        action = torch.cat([pressure, suction], dim=-1)
        return action

    def evaluate(self, obs):
        x = self.shared(obs)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mu, std)
        pressure_raw = normal.rsample()
        pressure = torch.tanh(pressure_raw)

        log_prob = normal.log_prob(pressure_raw) - torch.log(1 - pressure.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        suction = self.suction_head(x)
        action = torch.cat([pressure, suction], dim=-1)
        return action, log_prob

    def load_from_bc(self, path, device="cpu"):
        bc_state = torch.load(path, map_location=device)
        mapped_state = {}
        
        # Map keys manually
        for k, v in bc_state.items():
            if k.startswith("shared."):
                mapped_key = "shared." + k  # maps to shared.shared.*
            elif k.startswith("pressure_head.0."):
                mapped_key = k.replace("pressure_head.0.", "mu.")
            elif k.startswith("suction_head."):
                mapped_key = k.replace("suction_head.", "suction_head.0.")
            else:
                continue
            mapped_state[mapped_key] = v

        own_state = self.state_dict()
        loaded, skipped = [], []
        for k in own_state:
            if k in mapped_state and mapped_state[k].shape == own_state[k].shape:
                own_state[k] = mapped_state[k]
                loaded.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(own_state)
        print(f"[INFO] Loaded BC weights from {path}")
        print(f"[✓] Loaded {len(loaded)} keys:")
        for k in loaded:
            print(f"  ✓ {k}")
        print(f"[!] Skipped {len(skipped)} keys:")
        for k in skipped:
            print(f"  ✗ {k}")

# --- Wrapped Policy for SB3 ---
class MultiHeadWrappedPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, actor_class):
        super().__init__(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        self.actor_net = actor_class(obs_dim)
        self._is_training = True

    def forward(self, obs, deterministic=False):
        action = self.actor_net(obs, deterministic=deterministic)
        return action

    def _predict(self, obs, deterministic=False):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).to(next(self.parameters()).device)
            action = self.actor_net(x, deterministic=deterministic)
            return action

    def predict(self, obs, deterministic=False):
        action_tensor = self._predict(obs, deterministic)
        return action_tensor.cpu().numpy()

    def set_training_mode(self, mode: bool):
        self._is_training = mode
        self.actor_net.train(mode)

    def parameters(self):
        return self.actor_net.parameters()

    def to(self, device):
        self.actor_net.to(device)
        return self

    def load_from_bc(self, path, device="cpu"):
        self.actor_net.load_from_bc(path, device=device)

    def action_log_prob(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).to(next(self.parameters()).device)
        action, log_prob = self.actor_net.evaluate(x)
        return action, log_prob


# --- Custom SAC Policy ---
class MultiHeadSACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        self.custom_actor_class = MultiHeadActor
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_actor(self, features_extractor=None):
        return MultiHeadWrappedPolicy(self.observation_space, self.action_space, self.custom_actor_class)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.actor_target = self.make_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())


# --- Custom SAC Wrapper ---
from stable_baselines3.sac.sac import SAC as BaseSAC

class MultiHeadSAC(BaseSAC):
    policy_aliases = {
        "MultiHeadSACPolicy": MultiHeadSACPolicy
    }

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)


# --- Example usage ---
if __name__ == "__main__":
    from rl.utils.reward_logger_callback import RewardLoggerCallback
    from stable_baselines3.common.callbacks import CallbackList
    from rl.scripts.tasq_env import TASQEnv
    from rl.utils.training_progress_callback import TrainingProgressCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
    import multiprocessing

    def make_env(rank):
        def _init():
            return TASQEnv(urdf_path="rl/urdfs/TASQ_n7.urdf", no_links=7, gui=False)
        return _init

    num_cpu = multiprocessing.cpu_count()  # or choose a subset, e.g. 8

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
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
    bc_weights_path = "rl/models/tasq_bc/2_heads/tasq_bc_ep100_20250620-002326.pt"
    model.policy.actor.load_from_bc(bc_weights_path, device=model.device)
    model.policy.actor_target.load_state_dict(model.policy.actor.state_dict())

    # Define how many total timesteps
    total_timesteps = 5_000_000
    
    reward_logger = RewardLoggerCallback(log_interval=10_000, verbose=1)
    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps, log_interval=10_000)

    callback = CallbackList([reward_logger, progress_callback])

    # Start training
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("rl/models/tasq_sac/tasq_sac_multihead_2")

import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import CallbackList
from gym import spaces

from rl.models.actor import BCActor
from rl.utils.callbacks import RewardLoggerCallback, TrainingProgressCallback
from rl.envs.tasq_env import TASQEnv  # <- replace with your env import

# --- Custom Actor with BC weight loading ---
class SACActorWithBC(nn.Module):
    def __init__(self, sac_actor, obs_dim, act_dim, device):
        super().__init__()
        self.sac_actor = sac_actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

    def load_from_bc(self, bc_weights_path, device):
        print(f"Loading BC weights from {bc_weights_path}")
        bc_model = BCActor(self.obs_dim).to(device)
        bc_model.load_state_dict(torch.load(bc_weights_path, map_location=device))
        bc_model.eval()

        # Copy weights from BC model into SAC actor
        with torch.no_grad():
            for sac_layer, bc_layer in zip(self.sac_actor.mu.layers, bc_model.layers):
                if isinstance(sac_layer, nn.Linear) and isinstance(bc_layer, nn.Linear):
                    sac_layer.weight.copy_(bc_layer.weight)
                    sac_layer.bias.copy_(bc_layer.bias)

        print("✅ SAC actor initialized with BC weights")

# --- Custom SAC Policy with BC support ---
class BCInitializedSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_bc_weights(self, bc_weights_path, device):
        # Wrap actor with BC initialization logic
        sac_actor_wrapper = SACActorWithBC(
            sac_actor=self.actor,
            obs_dim=self.observation_space.shape[0],
            act_dim=self.action_space.shape[0],
            device=device,
        )
        sac_actor_wrapper.load_from_bc(bc_weights_path, device)

        # Sync target actor
        self.actor_target.load_state_dict(self.actor.state_dict())

# --- Main Training ---
if __name__ == "__main__":
    env = TASQEnv()  # or make_vec_env if you want parallel envs

    # Create SAC model with custom policy
    model = SAC(
        policy=BCInitializedSACPolicy,
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # --- Optional: Load pretrained BC weights ---
    bc_weights_path = "rl/models/tasq_bc/minimal_viable/..."
    model.policy.load_bc_weights(bc_weights_path, device=model.device)

    # --- Training ---
    total_timesteps = 5_000_000
    reward_logger = RewardLoggerCallback(log_interval=10_000, verbose=1)
    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps, log_interval=10_000)
    callback = CallbackList([reward_logger, progress_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("rl/models/tasq_sac/minimal_viable")

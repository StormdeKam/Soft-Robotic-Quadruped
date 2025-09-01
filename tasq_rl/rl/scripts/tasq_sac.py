# parallel_sac_train.py
import os
import torch
import torch.nn as nn
from gym import spaces
from functools import partial
from typing import Callable, List

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback


from rl.models.actor import BCActor
from rl.utils.callbacks.reward_logger_callback import RewardLoggerCallback
from rl.utils.callbacks.training_progress_callback import TrainingProgressCallback
from rl.scripts.tasq_env import TASQEnv  # your env

# ---------- BC-initialized Actor/Policy ----------
class SACActorWithBC(nn.Module):
    def __init__(self, sac_actor, obs_dim, act_dim, device):
        super().__init__()
        self.sac_actor = sac_actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

    def load_from_bc(self, bc_weights_path, device):
        print(f"[BC] Loading BC weights from {bc_weights_path}")
        bc_model = BCActor(self.obs_dim).to(device)
        state_dict = torch.load(bc_weights_path, map_location=device)
        bc_model.load_state_dict(state_dict)
        bc_model.eval()

        # Copy linear -> linear (keeps your original assumption)
        with torch.no_grad():
            for sac_layer, bc_layer in zip(self.sac_actor.mu.layers, bc_model.layers):
                if isinstance(sac_layer, nn.Linear) and isinstance(bc_layer, nn.Linear):
                    sac_layer.weight.copy_(bc_layer.weight)
                    sac_layer.bias.copy_(bc_layer.bias)

        print("[BC] ✅ SAC actor initialized with BC weights")

class BCInitializedSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_bc_weights(self, bc_weights_path, device):
        wrapper = SACActorWithBC(
            sac_actor=self.actor,
            obs_dim=self.observation_space.shape[0],
            act_dim=self.action_space.shape[0],
            device=device,
        )
        wrapper.load_from_bc(bc_weights_path, device)
        # keep target in sync
        self.actor_target.load_state_dict(self.actor.state_dict())

# ---------- VecEnv factory ----------
NO_LINKS_LIST = [5, 6, 7]
URDF_BY_LINKS = {
    5: "rl/urdfs/TASQ_n5.urdf",
    6: "rl/urdfs/TASQ_n6.urdf",
    7: "rl/urdfs/TASQ_n7.urdf",
}

def make_env_fn(rank: int, seed: int = 0, frame_stack: int = 4, gui: bool = False) -> Callable:
    """
    Returns a thunk that creates a single TASQEnv instance.
    We randomize no_links/URDF per worker at creation time.
    """
    def _init():
        import numpy as np
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)

        no_links = int(np.random.choice(NO_LINKS_LIST))
        urdf_path = URDF_BY_LINKS[no_links]

        # gui must be False for SubprocVecEnv; render with separate eval env if desired
        env = TASQEnv(
            urdf_path=urdf_path,
            no_links=no_links,
            gui=gui,
            render_mode=None,
            frame_stack=frame_stack,
        )
        env.seed(seed + rank)
        return env
    return _init

def make_vec_env(n_envs: int, seed: int = 0, use_subproc: bool = True, frame_stack: int = 4):
    env_fns: List[Callable] = [make_env_fn(rank=i, seed=seed, frame_stack=frame_stack, gui=False) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns) if use_subproc and n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec)  # keeps per-env episode stats for callbacks
    return vec

# ---------- Main ----------
if __name__ == "__main__":
    # --- Repro ---
    seed = 42
    set_random_seed(seed)

    # --- Parallelism ---
    # Tip: start with n_envs ~ #physical cores/2 if sim is heavy; adjust based on CPU usage.
    n_envs = max(1, min(os.cpu_count() or 1, 8))  # cap at 8 by default
    env = make_vec_env(n_envs=n_envs, seed=seed, use_subproc=True, frame_stack=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {n_envs} parallel envs on device: {device}")

    # --- SAC model (custom policy) ---
    model = SAC(
        policy=BCInitializedSACPolicy,
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        # With vec envs, this means "every env step"
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        device=device,
        seed=seed,
    )

    # --- Optional: load BC weights into policy ---
    bc_weights_path = "rl/models/tasq_bc/minimal_viable_V2/tasq_bc_ep150_20250901-133944.pt"
    if os.path.isfile(bc_weights_path):
        model.policy.load_bc_weights(bc_weights_path, device=model.device)
    else:
        print(f"[BC] Skipping: file not found at {bc_weights_path}")

    # --- Callbacks ---
    total_timesteps = 1_000_000
    log_interval = 10_000

    reward_logger = RewardLoggerCallback(log_interval=log_interval, verbose=1)
    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps, log_interval=log_interval)

    # Save a checkpoint every 100k steps (adjust as needed)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // n_envs,   # divide by n_envs because SB3 counts *per-env steps*
        save_path="rl/models/tasq_sac/checkpoints",
        name_prefix="tasq_sac",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    callback = CallbackList([reward_logger, progress_callback, checkpoint_callback])

    # --- Learn ---
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # --- Save ---
    out_dir = "rl/models/tasq_sac/minimal_viable"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    model.save(out_dir)
    print(f"[SAVE] Model saved to: {out_dir}")

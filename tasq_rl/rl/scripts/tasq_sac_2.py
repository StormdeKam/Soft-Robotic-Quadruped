import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List
from rl.utils.framestacking.dataset_stacker import StackedDataset
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

from rl.models.architecture.actor import BCActor
from rl.utils.callbacks.reward_logger_callback import RewardLoggerCallback
from rl.utils.callbacks.training_progress_callback import TrainingProgressCallback
from rl.scripts.tasq_env import TASQEnv
from rl.scripts.tasq_test import test_model, test_model_video_cv2


SAVE_PATH = "rl/models/tasq_sac/minimal_viable_V7/trial_1(9_25_2025)"
os.makedirs(SAVE_PATH, exist_ok=True)
# Observed state indices (skipping base_pos and base_vel)
OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base_angular_velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:35,  # phase
    35:39   # goal_pos and orientation
]

# Make initial actor LR tiny
initial_actor_lr = 1e-6  # effectively frozen
# We'll find the actor-related optimizer(s) and reduce LR below
# After creating the model, adjust optimizer param groups:
def set_optimizer_lr_for_actor(model, lr):
    # find all torch.optim.Optimizer objects in model.__dict__
    for k, v in model.__dict__.items():
        if isinstance(v, torch.optim.Optimizer):
            # adjust any param_group that contains parameters from model.policy.actor
            for pg in v.param_groups:
                # check if this param_group contains an actor param by identity
                params = pg.get("params", [])
                if any(p in list(model.policy.actor.parameters()) for p in params):
                    pg["lr"] = lr

# ---------------- Extended checkpoint callback ----------------
class ActorCheckpointCallback(CheckpointCallback):
    """
    Extends CheckpointCallback to also save the SAC actor weights (.pt).
    """
    def __init__(self, *args, actor_save_path: str = None, eval_runs=10, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_save_path = actor_save_path or self.save_path
        os.makedirs(self.actor_save_path, exist_ok=True)
        self.eval_runs = eval_runs
        self.device = device
        self.metrics_file = os.path.join(self.actor_save_path, "metrics.txt")

    def _on_step(self) -> bool:
        # Call parent to save the usual SB3 checkpoint
        super()._on_step()

        # Only save actor if checkpointing condition is met
        if self.n_calls % self.save_freq == 0:
            steps = self.num_timesteps
            actor_file = os.path.join(
                self.actor_save_path, f"{self.name_prefix}_actor_{steps}_steps.pt"
            )
            torch.save(self.model.policy.actor.state_dict(), actor_file)
            if self.verbose > 0:
                print(f"[ActorCheckpoint] ✅ Saved actor weights to {actor_file}")

                        # ----- Run evaluation -----
            print(f"[Eval] Running {self.eval_runs} test runs for {actor_file}")
            results = test_model(actor_file, device=self.device, n_runs=self.eval_runs)

            # Save metrics
            with open(self.metrics_file, "a") as f:
                f.write(f"Steps {steps}: {results}\n")

            # ----- Run video generation -----
            video_file = os.path.join(
                self.actor_save_path, f"test_run_{steps}_steps.mp4"
            )
            print(f"[Eval] Saving test video to {video_file}")
            test_model_video_cv2(actor_file, device=self.device, steps=1400*2, video_path=video_file)

        return True

# ---------------- Warm start callback ----------------
class WarmStartSACCallback(BaseCallback):
    def __init__(self, freeze_steps: int = 500_000, start_log_std: float = -6.0, end_log_std: float = -2.0, start_actor_lr: float = 1e-6, end_actor_lr: float = 3e-5, verbose: int = 0):
        super().__init__(verbose)
        self.freeze_steps = freeze_steps
        self.start = start_log_std
        self.end = end_log_std
        self.start_actor_lr = start_actor_lr
        self.end_actor_lr = end_actor_lr

    def _on_step(self) -> bool:
        num_steps = self.num_timesteps
        t = min(1.0, num_steps / max(1, self.freeze_steps))
        new_log_std = self.start + t * (self.end - self.start)
        # If policy.actor has a log_std parameter (SB3 Actor uses a network), attempt to set it:
        if hasattr(self.model.policy, "actor"):
            actor = self.model.policy.actor
            # Try to set any attribute named "log_std" if exists (rare)
            if hasattr(actor, "log_std") and isinstance(actor.log_std, torch.nn.Parameter):
                actor.log_std.data.fill_(new_log_std)

        # Ramp optimizer LR for the actor
        new_actor_lr = self.start_actor_lr + t * (self.end_actor_lr - self.start_actor_lr) if hasattr(self, "end_actor_lr") else self.start_actor_lr
        # robust way: iterate all optimizers and update param groups that contain actor params
        actor_params = set(self.model.policy.actor.parameters())
        for k, opt in self.model.__dict__.items():
            if isinstance(opt, torch.optim.Optimizer):
                for pg in opt.param_groups:
                    if any(p in actor_params for p in pg.get("params", [])):
                        pg["lr"] = new_actor_lr

        return True

# ---------------- BC-initialized Actor ----------------
class SACActorWithBC(nn.Module):
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.bc_actor = BCActor(obs_dim).to(device)
        self.bc_actor.eval()

        # Exploration noise control
        self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * -6.0)  # very small initially

    def load_from_bc(self, bc_weights_path):
        state_dict = torch.load(bc_weights_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("bc_actor."):
                new_state_dict[k[len("bc_actor."):]] = v
            else:
                new_state_dict[k] = v
        self.bc_actor.load_state_dict(new_state_dict)
        self.bc_actor.eval()

    def forward(self, obs, deterministic=False):
        """
        obs: [batch_size, obs_dim]
        deterministic: if True, return mean (BC output), else sample with log_std
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)

        # Get BC actor output
        pressures, suctions = self.bc_actor(obs)
        mu = torch.cat([pressures, suctions], dim=-1)

        if deterministic:
            # During evaluation or warm-start: just return BC action
            action = mu
        else:
            # Stochastic: add Gaussian noise with current log_std
            std = torch.exp(self.log_std)
            noise = torch.randn_like(mu) * std
            action = mu + noise

        return torch.clamp(action, -1.0, 1.0)

    def set_training_mode(self, mode: bool):
        if mode:
            self.bc_actor.train()
        else:
            self.bc_actor.eval()

# ---------------- Policy wrapper ----------------
class BCInitializedSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_bc_weights(self, bc_weights_path, device="cpu"):
        self._build(lr_schedule=lambda _: 3e-4)
        bc_wrapper = SACActorWithBC(
            obs_dim=self.observation_space.shape[0],
            act_dim=self.action_space.shape[0],
            device=device
        )
        bc_wrapper.load_from_bc(bc_weights_path)
        self.actor = bc_wrapper
        print("[BC] ✅ BC-initialized SAC actor loaded and ready.")

# ---------------- VecEnv ----------------
NO_LINKS_LIST = [5, 6, 7]
URDF_BY_LINKS = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}

def make_env_fn(rank: int, seed: int = 0, frame_stack: int = 4, gui: bool = False) -> Callable:
    def _init():
        import numpy as np
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        no_links = int(np.random.choice(NO_LINKS_LIST))
        urdf_path = URDF_BY_LINKS[no_links]
        env = TASQEnv(urdf_path=urdf_path, no_links=no_links, gui=gui, frame_stack=frame_stack)
        env.seed(seed + rank)
        return env
    return _init

def make_vec_env(n_envs: int, seed: int = 0, use_subproc: bool = True, frame_stack: int = 4):
    env_fns: List[Callable] = [make_env_fn(rank=i, seed=seed, frame_stack=frame_stack, gui=False) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns) if use_subproc and n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec)
    return vec

# ---------------- Main ----------------
if __name__ == "__main__":
    seed = 42
    set_random_seed(seed)
    n_envs = max(1, min(os.cpu_count() or 1, 8))
    env = make_vec_env(n_envs=n_envs, seed=seed, use_subproc=True, frame_stack=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {n_envs} parallel envs on device: {device}")

    model = SAC(
        policy=BCInitializedSACPolicy,
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef=0.0001,  # smaller entropy
        target_entropy=-16,  # less negative than default (-action_dim)
        verbose=1,
        device=device,
        seed=seed,
    )

    # Load BC weights
    bc_weights_path = "rl/models/tasq_bc/minimal_viable_V7/tasq_bc_ep100.pt"
    
    if os.path.isfile(bc_weights_path):
        model.policy.load_bc_weights(bc_weights_path, device=model.device)
    else:
        print(f"[BC] Skipping: file not found at {bc_weights_path}")

    # CALL THIS IMMEDIATELY AFTER model = SAC(...)
    set_optimizer_lr_for_actor(model, initial_actor_lr)
    print(f"[BC] set actor LR to {initial_actor_lr}")

    # Freeze actor params initially
    for p in model.policy.actor.parameters():
        p.requires_grad = False

    # Reduce actor LR after init
    if hasattr(model.policy, "actor_optimizer"):
        for g in model.policy.actor_optimizer.param_groups:
            g["lr"] = 3e-5

    # Gradient clipping (actor)
    #if hasattr(model.policy, "actor_optimizer"):
        #torch.nn.utils.clip_grad_norm_(model.policy.actor.parameters(), 0.5)

    # Callbacks
    total_timesteps = 10_000_000  
    log_interval = 10_000
    reward_logger = RewardLoggerCallback(log_interval=log_interval, verbose=1)
    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps, log_interval=log_interval)
    checkpoint_callback = ActorCheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path= os.path.join(SAVE_PATH,"checkpoints"),
        name_prefix="tasq_sac",
        save_replay_buffer=True,
        save_vecnormalize=True,
        actor_save_path=os.path.join(SAVE_PATH,"actors"), 
        device=device,
        eval_runs=10,
        verbose=1,
    )

    warm_start_callback = WarmStartSACCallback(freeze_steps=500_000, verbose=1)
    callback = CallbackList([reward_logger, progress_callback, checkpoint_callback, warm_start_callback])
    
    # ---------- Seed Replay Buffer with BC Demos ----------
    states_path = "rl/datasets/minimal_viable_v7/demo_states_merged.npy"
    actions_path = "rl/datasets/minimal_viable_v7/demo_actions_merged.npy"
    frame_stack = getattr(env, "frame_stack", 4)  # match TASQEnv setting

    if os.path.isfile(states_path) and os.path.isfile(actions_path):
        # Create stacked dataset
        demo_dataset = StackedDataset(
            state_path=states_path,
            action_path=actions_path,
            N=frame_stack,
            observed_indices=OBSERVED_INDICES
        )

        n_envs = model.n_envs
        max_seed_transitions = min(len(demo_dataset), model.replay_buffer.buffer_size // 2)  # up to 50% of buffer
        transitions_added = 0
        perm = np.random.permutation(len(demo_dataset) - 1) 

        for i in perm:
            s, a = demo_dataset[i]
            s2, _ = demo_dataset[i + 1]

            # Convert to numpy
            s = s.numpy()[None, :]    # shape (1, obs_dim)
            a = a.numpy()[None, :]    # shape (1, action_dim)
            s2 = s2.numpy()[None, :]  # shape (1, obs_dim)

            # Repeat for n_envs
            s_batch = np.repeat(s, n_envs, axis=0)
            a_batch = np.repeat(a, n_envs, axis=0)
            s2_batch = np.repeat(s2, n_envs, axis=0)
            r_batch = np.zeros(n_envs, dtype=np.float32)
            d_batch = np.zeros(n_envs, dtype=bool)
            infos_batch = [{} for _ in range(n_envs)]

            model.replay_buffer.add(
                obs=s_batch,
                next_obs=s2_batch,
                action=a_batch,
                reward=r_batch,
                done=d_batch,
                infos=infos_batch
            )

            transitions_added += 1
            if transitions_added >= max_seed_transitions:
                break

        print(f"[BC] ✅ Seeded replay buffer with {transitions_added} stacked demo transitions (repeated for {n_envs} envs).")
    else:
        print("[BC] ⚠️ Demo dataset not found, skipping replay buffer seeding.")
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    model.save(os.path.join(SAVE_PATH, "tasq_sac_full"))
    torch.save(model.policy.actor.state_dict(), os.path.join(SAVE_PATH, "tasq_sac_actor.pt"))
    print(f"[SAVE] SAC model + actor saved to {SAVE_PATH}")

# rl/scripts/tasq_sac_seeded.py
import os
import types
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple

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

# ---------------- Paths / config ----------------
SAVE_PATH = "rl/models/tasq_sac/low_freqv6/sac2/trial1_04_11_25"
os.makedirs(SAVE_PATH, exist_ok=True)

# Where you saved the reward-filled replay buffer (from the script I provided)
#REPLAY_BUFFER_PATH = "datasets/replay_buffers/low_freq_v6/replay_buffer3.npz"

# Training / schedule sizes (total = 200_000)
WARM_START_STEPS = 20_000
EXPLORATION_STEPS = 100_000
DECAY_STEPS = 0#50_000
TOTAL_SCHEDULE_STEPS = WARM_START_STEPS + EXPLORATION_STEPS + DECAY_STEPS
TOTAL_TIMESTEPS = TOTAL_SCHEDULE_STEPS  # train for exactly the schedule length (tweak as desired)

# Actor LR schedule endpoints (warm -> mid -> decay-end)
START_ACTOR_LR = 2e-3
MID_ACTOR_LR = 1e-4
END_ACTOR_LR = 1e-6

CRITIC_LR = 2e-3


# Target entropy schedule (will be adjusted during decay to favor exploitation)
# We'll pick a reasonable mid target and make the end target *more negative* to reduce entropy target.
ENTROPY_TARGET_MID_SCALE = -1.0  # target = -0.5 * action_dim (your previous choice)
ENTROPY_TARGET_END_FACTOR = 2.0  # end_target = mid_target * ENTROPY_TARGET_END_FACTOR (more negative -> lower entropy desired)

# Other SAC hyperparams
BUFFER_SIZE = 10_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.97
TRAIN_FREQ = 3
GRADIENT_STEPS = 1
START_LOG_STD = -0.5 

# Vectorized env config
NO_LINKS_LIST = [5, 6, 7]
URDF_BY_LINKS = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}

# ---------------- Multi-phase actor lr schedule (unused, but here for reference) ----------------
def multiphase_lr_schedule(
    total_steps: int,
    warm_start_steps: int = 50_000,
    exploration_steps: int = 150_000,
    decay_steps: int = 50_000,
    start_lr: float = 2e-3,
    mid_lr: float = 1e-5,
    end_lr: float = 1e-7,
):
    """
    Returns an SB3-compatible learning rate schedule:
    learning_rate(progress_remaining: float) -> float
    
    SB3 calls this with progress_remaining in [1, 0],
    so we map it back to the current step internally.
    """

    def lr_fn(progress_remaining: float) -> float:
        # Convert progress_remaining (1→0) to step (0→total_steps)
        step = int((1 - progress_remaining) * total_steps)

        if step <= warm_start_steps:
            t = min(1.0, step / max(1, warm_start_steps))
            smooth_t = t**2# 1 - np.exp(-5 * t)
            lr = start_lr + smooth_t * (mid_lr - start_lr)
        elif step <= warm_start_steps + exploration_steps:
            exploration_step = step - warm_start_steps 
            t = min(1.0, exploration_step / max(1, exploration_steps))
            smooth_t = t**2#1 - np.exp(-5 * t)
            lr = mid_lr + smooth_t * (end_lr - mid_lr)
        else:
            # Smooth decay from mid_lr → end_lr
            decay_step = step - (warm_start_steps + exploration_steps)
            t = min(1.0, decay_step / max(1, decay_steps))
            smooth_t = 1 - np.exp(-5 * t)
            lr = mid_lr + smooth_t * (end_lr - mid_lr)

        return lr

    return lr_fn



# ---------------- Extended checkpoint callback (unchanged) ----------------
class ActorCheckpointCallback(CheckpointCallback):
    def __init__(self, *args, actor_save_path: str = None, eval_runs=3, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_save_path = actor_save_path or self.save_path
        os.makedirs(self.actor_save_path, exist_ok=True)
        self.eval_runs = eval_runs
        self.device = device
        self.metrics_file = os.path.join(self.actor_save_path, "metrics.txt")

    def _on_step(self) -> bool:
        super()._on_step()
        if self.n_calls % self.save_freq == 0:
            steps = self.num_timesteps
            actor_file = os.path.join(self.actor_save_path, f"{self.name_prefix}_actor_{steps}_steps.pt")
            torch.save(self.model.policy.actor.state_dict(), actor_file)
            if self.verbose > 0:
                print(f"[ActorCheckpoint] ✅ Saved actor weights to {actor_file}")
            print(f"[Eval] Running {self.eval_runs} test runs for {actor_file}")
            results = test_model(actor_file, device=self.device, n_runs=self.eval_runs)
            with open(self.metrics_file, "a") as f:
                f.write(f"Steps {steps}: {results}\n")
        return True

# ---------------- Multi-phase exploration callback (modified) ----------------
class MultiPhaseExplorationCallback(BaseCallback):
    """
    Implements warmstart -> exploration -> decay schedule.
    - DOES NOT force-fill actor.log_std after initialization (log_std remains trainable).
    - Adjusts actor optimizer LR across phases.
    - Adjusts model.target_entropy across decay phase (works when ent_coef='auto').
    """
    def __init__(self,
                 warm_start_steps: int = WARM_START_STEPS,
                 fixed_steps: int = EXPLORATION_STEPS,
                 decay_steps: int = DECAY_STEPS,
                 start_actor_lr: float = START_ACTOR_LR,
                 mid_actor_lr: float = MID_ACTOR_LR,
                 end_actor_lr: float = END_ACTOR_LR,
                 mid_target_entropy_scale: float = ENTROPY_TARGET_MID_SCALE,
                 end_target_factor: float = ENTROPY_TARGET_END_FACTOR,
                 verbose: int = 1):
        super().__init__(verbose)

                # === Logging setup ===
        self.temp_log_path = os.path.join(SAVE_PATH, "temperature_log.csv")
        self.entropy_log_path = os.path.join(SAVE_PATH, "entropy_log.csv")

        # Initialize CSV files with headers if not existing
        if not os.path.exists(self.temp_log_path):
            with open(self.temp_log_path, "w") as f:
                f.write("timestep,temperature\n")
        if not os.path.exists(self.entropy_log_path):
            with open(self.entropy_log_path, "w") as f:
                f.write("timestep,entropy\n")

        self._last_entropy_log = 0
        self._log_every = 1000  # log every 1000 timesteps (adjust as desired)

        self.log_std_file = os.path.join(SAVE_PATH, "log_std_log.csv")
        with open(self.log_std_file, "w") as f:
            f.write("step,log_std_mean,log_std_min,log_std_max\n")

        self.warm_start_steps = warm_start_steps
        self.fixed_steps = fixed_steps
        self.decay_steps = decay_steps
        self.total_steps = warm_start_steps + fixed_steps + decay_steps

        self.start_actor_lr = start_actor_lr
        self.mid_actor_lr = mid_actor_lr
        self.end_actor_lr = end_actor_lr

        self.mid_target_entropy_scale = mid_target_entropy_scale
        self.end_target_factor = end_target_factor

    def _set_requires_grad(self, model, requires_grad: bool):
        """Freeze/unfreeze actor parameters only."""
        actor = getattr(model.policy, "actor", None)
        if actor is not None:
            for p in actor.parameters():
                p.requires_grad_(requires_grad)
        if self.verbose:
            state = "frozen" if not requires_grad else "unfrozen"
            print(f"[ExplorationSchedule] Actor {state}.")

    def _log_temperature(self, step):
        """Log entropy temperature α to CSV."""
        if hasattr(self.model, "ent_coef_tensor"):
            alpha = float(self.model.ent_coef_tensor.item())
        elif hasattr(self.model, "log_ent_coef"):
            alpha = float(torch.exp(self.model.log_ent_coef.detach()).item())
        else:
            alpha = float(self.model.ent_coef)
        with open(self.temp_log_path, "a") as f:
            f.write(f"{step},{alpha:.6f}\n")

    def _log_entropy(self, step):
        """Estimate and log current policy entropy."""
        obs = torch.as_tensor(
            self.model._last_obs, device=self.model.device, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            dist = self.model.policy.actor.get_action_dist_from_obs(obs)
            entropy = float(dist.entropy().mean().item())
        with open(self.entropy_log_path, "a") as f:
            f.write(f"{step},{entropy:.6f}\n")

    def _on_training_start(self) -> None:
        # Set initial actor.log_std (just once) if available, but keep trainable
        if hasattr(self.model.policy, "actor") and hasattr(self.model.policy.actor, "log_std"):
            actor = self.model.policy.actor
            if isinstance(actor.log_std, torch.nn.Parameter):
                # initialize to a low log_std but let it be trainable afterwards
                with torch.no_grad():
                    actor.log_std.data.fill_(START_LOG_STD)
                if self.verbose:
                    print(f"[ExplorationSchedule] Initialized actor.log_std to {START_LOG_STD} (frozen).")
        else:
            print("[ExplorationSchedule] Model policy has no actor.log_std attribute; skipping initialization.")

        # Set initial target entropy (mid) if ent_coef is trainable ("auto")
        action_dim = int(self.model.action_space.shape[0])
        mid_target = float(self.mid_target_entropy_scale * action_dim)
        # store mid_target for schedule
        self.mid_target = mid_target
        self.end_target = mid_target * self.end_target_factor
        # If model has target_entropy attribute, set it initially to mid_target
        if hasattr(self.model, "target_entropy"):
            try:
                self.model.target_entropy = float(mid_target)
                if self.verbose:
                    print(f"[ExplorationSchedule] Set initial target_entropy = {self.model.target_entropy:.3f}")
            except Exception:
                pass
        else:
            print("[ExplorationSchedule] Model has no target_entropy attribute; skipping initial set.")


        # Freeze actor and ent_coef for warmstart
        self._set_requires_grad(self.model, requires_grad=False)

        # ✅ Keep log_std trainable
        #"""
        actor = getattr(self.model.policy, "actor", None)
        if actor is not None and hasattr(actor, "log_std"):
            actor.log_std.requires_grad_(True)
            if self.verbose:
                print("[ExplorationSchedule] log_std kept trainable during warmstart.")
        #"""
        # Keep entropy coefficient trainable (to avoid SB3 crash)
        if hasattr(self.model, "log_ent_coef"):
            self.model.log_ent_coef.requires_grad_(True)
        else:
            print("[ExplorationSchedule] Model has no log_ent_coef attribute; skipping ent_coef freeze.")


    def _on_step(self) -> bool:
        step = self.num_timesteps

        if step <= self.warm_start_steps:
            t = step / max(1, self.warm_start_steps)
            #actor_lr = self.start_actor_lr
            target_entropy = self.mid_target
            #self._set_requires_grad(self.model, requires_grad=False)  # <--- freeze
            # Disable entropy coefficient optimizer step while ent_coef is frozen
            #if hasattr(self.model, "ent_coef_optimizer"):
                #for g in self.model.ent_coef_optimizer.param_groups:
                    #g["lr"] = 0.0

        elif step <= self.warm_start_steps + self.fixed_steps:
            if not getattr(self, "_unfrozen_actor", False) and step >= self.warm_start_steps:
                self._set_requires_grad(self.model, requires_grad=True)
                actor = getattr(self.model.policy, "actor", None)
                if actor is not None and hasattr(actor, "log_std"):
                    actor.log_std.requires_grad_(True)
                print(f"[ExplorationSchedule] 🔓 Warmstart complete — actor and ent_coef unfrozen.")
                self._unfrozen_actor = True                

            target_entropy = self.mid_target
        else:
            decay_step = step - (self.warm_start_steps + self.fixed_steps)
            t = min(1.0, decay_step / max(1, self.decay_steps))
            smooth_t = 1 - np.exp(-5 * t)
            target_entropy = self.mid_target + smooth_t * (self.end_target - self.mid_target)

        # Apply target_entropy if model uses auto entropy (i.e., ent_coef is trainable)
        if hasattr(self.model, "target_entropy"):
            try:
                self.model.target_entropy = float(target_entropy)
            except Exception:
                pass

        # Optional logging
        if self.verbose and step % 1000 == 0:
            ent = getattr(self.model, "ent_coef", None)
            # if ent_coef is a tensor param, print its value
            ent_val = None
            try:
                if hasattr(self.model, "ent_coef_tensor"):
                    ent_val = float(self.model.ent_coef_tensor.detach().cpu())
                elif isinstance(ent, (float, int)):
                    ent_val = float(ent)
                elif hasattr(self.model, "ent_coef") and isinstance(self.model.ent_coef, torch.nn.Parameter):
                    ent_val = float(self.model.ent_coef.detach().cpu())
            except Exception:
                ent_val = None
            print(f"[ExplorationSchedule] step={step} | target_entropy={target_entropy:.3f} | ent_coef={ent_val}")

        if step % 1000 == 0:
            print("Critic LR:", [g['lr'] for g in model.critic.optimizer.param_groups])
            actor = getattr(self.model.policy, "actor", None)
            if actor is not None and hasattr(actor, "log_std"):
                log_std_vals = actor.log_std.detach().cpu().numpy()
                with open(self.log_std_file, "a") as f:
                    f.write(f"{step},{log_std_vals.mean():.6f},{log_std_vals.min():.6f},{log_std_vals.max():.6f}\n")

                # --- Logging ---
        if step - self._last_entropy_log >= self._log_every:
            self._log_temperature(step)
            #self._log_entropy(step)
            self._last_entropy_log = step

        actor = getattr(self.model.policy, "actor", None)
        if actor is not None and step % 1000 == 0:
            with torch.no_grad():
                grad_norm = 0.0
                for p in actor.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm().item()
                print(f"[ExplorationCheck] step={step} | total_grad_norm={grad_norm:.6f}")


        return True

# ---------------- Gradient-clipped SAC subclass ----------------
class BCClippedSAC(SAC):
    """
    Subclass of SAC that clips gradients for actor and critic updates.
    Integrates seamlessly with BCInitializedSACPolicy.
    """
    def __init__(self, *args, demo_sample_ratio=0.3, max_grad_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = max_grad_norm
        self.demo_sample_ratio = demo_sample_ratio
        self.demo_replay_buffer = None  # will hold frozen BC demos

    def _update_actor(self, actor_loss: torch.Tensor) -> None:
        self.actor.optimizer.zero_grad()
        actor_loss.backward()

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in self.actor.parameters() if p.grad is not None])
        )
        if total_norm > self.max_grad_norm:
            print(f"[GradClip] Actor grad norm before clip: {total_norm:.3f}")

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()

    def _update_critic(self, critic_loss: torch.Tensor) -> None:
        self.critic.optimizer.zero_grad()
        critic_loss.backward()

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in self.critic.parameters() if p.grad is not None])
        )
        print(f"[Warmstart Critic Step] grad norm {total_norm:.3f}")

        if total_norm > self.max_grad_norm:
            print(f"[GradClip] Critic grad norm before clip: {total_norm:.3f}")

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic.optimizer.step()

    def load_demo_replay_buffer(self, npz_path, max_transitions=None, verbose=True):
        """Load demo transitions into a separate frozen buffer."""
        import numpy as np
        from stable_baselines3.common.buffers import ReplayBuffer

        data = np.load(npz_path)
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        rewards = data["rewards"]
        dones = data["dones"]

        n_samples = len(states)
        if max_transitions is not None:
            n_samples = min(n_samples, max_transitions)

        demo_buffer = ReplayBuffer(
            buffer_size=n_samples,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=1,
        )

        for i in range(n_samples):
            demo_buffer.add(
                states[i], next_states[i], actions[i], rewards[i], dones[i], infos={}
            )

        self.demo_replay_buffer = demo_buffer
        if verbose:
            print(f"[DemoBuffer] Loaded {n_samples} demo transitions into frozen buffer.")
        return n_samples
    
    def _sample_replay_buffer(self, batch_size: int):
        """Sample partly from demo buffer (if available) and partly from online buffer."""
        if self.demo_replay_buffer is None or self.demo_sample_ratio <= 0:
            return self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        demo_bs = int(batch_size * self.demo_sample_ratio)
        online_bs = batch_size - demo_bs

        # Sample from both
        online_batch = self.replay_buffer.sample(online_bs, env=self._vec_normalize_env)
        demo_batch = self.demo_replay_buffer.sample(demo_bs, env=self._vec_normalize_env)

        # Concatenate batches along batch dimension
        from stable_baselines3.common.buffers import ReplayBufferSamples
        import torch

        obs = torch.cat([online_batch.observations, demo_batch.observations], dim=0)
        actions = torch.cat([online_batch.actions, demo_batch.actions], dim=0)
        next_obs = torch.cat([online_batch.next_observations, demo_batch.next_observations], dim=0)
        dones = torch.cat([online_batch.dones, demo_batch.dones], dim=0)
        rewards = torch.cat([online_batch.rewards, demo_batch.rewards], dim=0)

        return ReplayBufferSamples(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=rewards,
        )

    
# ---------------- BC-initialized actor wrapper (unchanged except we keep log_std trainable) ----------------
class SACActorWithBC(nn.Module):
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.bc_actor = BCActor(obs_dim).to(device)
        self.bc_actor.eval()
        self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * START_LOG_STD)

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

    def set_training_mode(self, mode: bool):
        if mode:
            self.bc_actor.train()
        else:
            self.bc_actor.eval()


    def forward(self, obs: torch.Tensor, deterministic=False) -> torch.Tensor:
        """Used by SB3 predict() → must return **action tensor only**"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)

        pressures, suctions = self.bc_actor(obs)
        mu = torch.cat([pressures, suctions], dim=-1)

        if deterministic:
            return torch.clamp(mu, -1.0, 1.0)

        std = torch.exp(self.log_std)
        noise = torch.randn_like(mu) * std
        return torch.clamp(mu + noise, -1.0, 1.0)
    
    def get_action_dist_from_obs(self, obs: torch.Tensor):
        """Compute distribution using existing get_action_and_log_prob() logic"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)

        pressures, suctions = self.bc_actor(obs)
        print("get_action_dist_from_obs:", pressures.shape, suctions.shape)  # <--- debu
        mu = torch.cat([pressures, suctions], dim=-1)
        std = torch.exp(self.log_std)
        std = std.expand_as(mu)  # Ensure shapes match

        return torch.distributions.Normal(mu, std)

    def get_action_and_log_prob(self, obs: torch.Tensor):
        """Separate method for SAC training → returns action, log_prob, mean"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)

        pressures, suctions = self.bc_actor(obs)
        mu = torch.cat([pressures, suctions], dim=-1)
        std = torch.exp(self.log_std)

        # Sample action
        noise = torch.randn_like(mu)
        action = mu + noise

        # Compute log_prob
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        mean_action = torch.clamp(mu, -1.0, 1.0)

        return torch.clamp(action, -1.0, 1.0), log_prob, mean_action

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SB3-compatible helper for SAC training"""
        action, log_prob, _ = self.get_action_and_log_prob(obs)
        return action, log_prob


# ---------------- Policy wrapper (unchanged) ----------------
class BCInitializedSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_bc_weights(self, bc_weights_path, device="cpu"):
        # build policy (so class attributes exist)
        self._build(lr_schedule=lambda _: 3e-4)
        bc_wrapper = SACActorWithBC(
            obs_dim=self.observation_space.shape[0],
            act_dim=self.action_space.shape[0],
            device=device
        )
        bc_wrapper.load_from_bc(bc_weights_path)
        self.actor = bc_wrapper
        print("[BC] ✅ BC-initialized SAC actor loaded and ready.")

        # Reinitialize actor optimizer to include bc actor params
        if hasattr(self, "actor_optimizer_class") and hasattr(self, "lr_schedule"):
            lr = self.lr_schedule(1)
            self.actor_optimizer = self.actor_optimizer_class(self.actor.parameters(), lr=lr)
            print(f"[BC] 🔁 Reinitialized actor optimizer with lr={lr}")
        else:
            import torch
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            print("[BC] ⚠️ Used fallback Adam optimizer for BC actor.")
        


# ---------------- VecEnv ----------------
def make_env_fn(rank: int, seed: int = 0, frame_stack: int = 4, gui: bool = False) -> Callable:
    def _init():
        import numpy as np, torch
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        no_links = 7#int(np.random.choice(NO_LINKS_LIST))
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

def seed_replay_buffer_from_npz(model, npz_path, verbose=True, max_transitions=None):
    """
    Load a replay buffer .npz file and seed SB3 replay buffer.

    Handles both flattened (N, dim) and vectorized (T, n_envs, dim) data.

    Parameters
    ----------
    model : stable_baselines3 model with replay_buffer
    npz_path : str
        Path to .npz file with keys: states, actions, next_states, rewards, dones
    max_transitions : int, optional
        Maximum number of transitions to add (default = all)
    """
    import numpy as np

    data = np.load(npz_path)
    required = ["states", "actions", "next_states", "rewards", "dones"]
    for k in required:
        if k not in data:
            raise ValueError(f"Missing key '{k}' in {npz_path}, found keys: {list(data.keys())}")

    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    rewards = data["rewards"]
    dones = data["dones"]

    # Determine shape
    if states.ndim == 3:
        T, n_envs, obs_dim = states.shape
    else:
        T = states.shape[0]
        n_envs = 1
        states = states[:, np.newaxis, :]
        actions = actions[:, np.newaxis, :]
        next_states = next_states[:, np.newaxis, :]
        rewards = rewards[:, np.newaxis]
        dones = dones[:, np.newaxis]

    total_transitions = T * n_envs
    if max_transitions is None:
        max_transitions = total_transitions

    added = 0
    for t in range(T):
        # batch for all envs at timestep t
        obs_t = states[t]
        action_t = actions[t]
        next_obs_t = next_states[t]
        reward_t = rewards[t]
        done_t = dones[t]
        infos_t = [{} for _ in range(n_envs)]

        # Only add if we haven't reached max_transitions
        if added + n_envs > max_transitions:
            # slice to fit
            slice_n = max_transitions - added
            obs_t = obs_t[:slice_n]
            action_t = action_t[:slice_n]
            next_obs_t = next_obs_t[:slice_n]
            reward_t = reward_t[:slice_n]
            done_t = done_t[:slice_n]
            infos_t = infos_t[:slice_n]

        model.replay_buffer.add(
            obs=obs_t,
            next_obs=next_obs_t,
            action=action_t,
            reward=reward_t,
            done=done_t,
            infos=infos_t
        )
        added += obs_t.shape[0]

        if verbose and (t+1) % 100 == 0:
            print(f"[Seed] Processed {min(added, max_transitions)}/{max_transitions} transitions")

        if added >= max_transitions:
            break

    if verbose:
        print(f"[Seed] Finished seeding. Added {added}/{max_transitions} transitions.")
    return added



# ---------------- Main ----------------
if __name__ == "__main__":
    seed = 44
    set_random_seed(seed)
    n_envs = max(1, min(os.cpu_count() or 1, 8))
    env = make_vec_env(n_envs=n_envs, seed=seed, use_subproc=True, frame_stack=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {n_envs} parallel envs on device: {device}")


    lr_schedule = multiphase_lr_schedule(
        total_steps=TOTAL_SCHEDULE_STEPS,
        warm_start_steps=WARM_START_STEPS,
        exploration_steps=EXPLORATION_STEPS,
        decay_steps=DECAY_STEPS,
        start_lr=START_ACTOR_LR,
        mid_lr=MID_ACTOR_LR,
        end_lr=END_ACTOR_LR
    )



    model = BCClippedSAC(
        policy=BCInitializedSACPolicy,
        env=env,
        learning_rate=lr_schedule, 
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        ent_coef='auto',
        target_entropy=-16.0,
        verbose=1,
        device=device,
        seed=seed,
        max_grad_norm=1.0,
        demo_sample_ratio=0.3,
    )

    # Load BC actor weights into policy (keeps log_std trainable)
    bc_weights_path = "rl/models/tasq_bc/low_freq_v6/tasq_bc_ep200.pt"
    if os.path.isfile(bc_weights_path):
        model.policy.load_bc_weights(bc_weights_path, device=model.device)
    else:
        print(f"[BC] Skipping: file not found at {bc_weights_path}")


    # Recreate actor optimizer to include log_std
    model.actor = model.policy.actor  # ensure SAC uses your BC actor reference
    model.actor.optimizer = torch.optim.Adam(model.policy.actor.parameters(), lr=START_ACTOR_LR)
    model.policy.actor_optimizer = model.actor.optimizer

    # Ensure critic uses a fixed learning rate while actor follows lr_schedule
    try:
        # Set critic optimizer LR to fixed value immediately
        for group in model.policy.critic.optimizer.param_groups:
            group['lr'] = CRITIC_LR

        # Monkey-patch policy _update_learning_rate to only update actor (and ent_coef) via schedule
        def _custom_update_learning_rate(self):
            current_lr = self.lr_schedule(self._current_progress_remaining)
            # Actor follows schedule
            self.set_learning_rate(self.actor.optimizer, current_lr)
            # Keep critic fixed
            for group in self.critic.optimizer.param_groups:
                group['lr'] = CRITIC_LR
            # Keep ent coef consistent with actor schedule if present
            if hasattr(self, 'log_ent_coef_optimizer') and self.log_ent_coef_optimizer is not None:
                self.set_learning_rate(self.log_ent_coef_optimizer, current_lr)

        model.policy._update_learning_rate = types.MethodType(_custom_update_learning_rate, model.policy)
    except Exception as e:
        print(f"[LR Setup] Warning: could not set separate critic LR: {e}")


    initial_ent_coef = 0.1
    if model.ent_coef == 'auto':
        with torch.no_grad():
            model.log_ent_coef.data.fill_(np.log(initial_ent_coef))

    # ---------- Seed replay buffer from saved reward-filled buffer ----------
    """
    if os.path.isfile(REPLAY_BUFFER_PATH):
        print(f"[Seed] Loading replay buffer from {REPLAY_BUFFER_PATH} ...")
        added = seed_replay_buffer_from_npz(model, REPLAY_BUFFER_PATH, verbose=True)
        print(f"[Seed] Added {added} transitions from {REPLAY_BUFFER_PATH}")
    else:
        print(f"[Seed] Replay buffer file not found: {REPLAY_BUFFER_PATH}. Skipping seeding.")

    if os.path.isfile(REPLAY_BUFFER_PATH):
        print(f"[Seed] Loading BC demo buffer from {REPLAY_BUFFER_PATH} ...")
        model.load_demo_replay_buffer(REPLAY_BUFFER_PATH, verbose=True)
    else:
        print(f"[Seed] Demo replay buffer not found at {REPLAY_BUFFER_PATH}. Skipping demo seeding.")
    """
    # ---------------- Callbacks ----------------
    total_timesteps = TOTAL_TIMESTEPS
    log_interval = 1000
    reward_log_path = os.path.join(SAVE_PATH, "reward_log.csv")
    reward_logger = RewardLoggerCallback(log_path=reward_log_path, log_interval=log_interval, verbose=1)
    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps, log_interval=log_interval)
    checkpoint_callback = ActorCheckpointCallback(
        save_freq=10_000 // max(1, n_envs),
        save_path=os.path.join(SAVE_PATH, "checkpoints"),
        name_prefix="tasq_sac",
        save_replay_buffer=True,
        save_vecnormalize=True,
        actor_save_path=os.path.join(SAVE_PATH, "actors"),
        device=device,
        eval_runs=3,
        verbose=1,
    )

    schedule_callback = MultiPhaseExplorationCallback(
        warm_start_steps=WARM_START_STEPS,
        fixed_steps=EXPLORATION_STEPS,
        decay_steps=DECAY_STEPS,
        start_actor_lr=START_ACTOR_LR,
        mid_actor_lr=MID_ACTOR_LR,
        end_actor_lr=END_ACTOR_LR,
        mid_target_entropy_scale=ENTROPY_TARGET_MID_SCALE,
        end_target_factor=ENTROPY_TARGET_END_FACTOR,
        verbose=1,
    )

    callback = CallbackList([reward_logger, progress_callback, checkpoint_callback, schedule_callback])

    # ---------- Learn ----------
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # ---------- Save ----------
    model.save(os.path.join(SAVE_PATH, "tasq_sac_full"))
    torch.save(model.policy.actor.state_dict(), os.path.join(SAVE_PATH, "tasq_sac_actor.pt"))
    print(f"[SAVE] SAC model + actor saved to {SAVE_PATH}")

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
SAVE_PATH = "rl/models/tasq_sac/low_freqv6/sac3/trial1_04_11_25"
os.makedirs(SAVE_PATH, exist_ok=True)

# Training / schedule sizes (total = 200_000)
WARM_START_STEPS = 20_000
EXPLORATION_STEPS = 100_000
TOTAL_SCHEDULE_STEPS = WARM_START_STEPS + EXPLORATION_STEPS
TOTAL_TIMESTEPS = TOTAL_SCHEDULE_STEPS  # train for exactly the schedule length (tweak as desired)

# Actor LR schedule endpoints (warm -> mid -> decay-end)
START_ACTOR_LR = 2e-3
MID_ACTOR_LR = 1e-5
END_ACTOR_LR = 1e-6

CRITIC_LR = 1e-3

ENTROPY_TARGET_MID_SCALE = -1.0

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

# ---------------- Multi-phase actor lr schedule  ----------------
def multiphase_lr_schedule(
    total_steps: int,
    warm_start_steps: int = 20_000,
    exploration_steps: int = 100_000,
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
        else:
            exploration_step = step - warm_start_steps 
            t = min(1.0, exploration_step / max(1, exploration_steps))
            smooth_t = t**2#1 - np.exp(-5 * t)
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
    Implements warmstart -> exploration schedule.
    - DOES NOT force-fill actor.log_std after initialization (log_std remains trainable).
    - Adjusts actor optimizer LR across phases.
    """
    def __init__(self,
                 warm_start_steps: int = WARM_START_STEPS,
                 fixed_steps: int = EXPLORATION_STEPS,
                 start_actor_lr: float = START_ACTOR_LR,
                 mid_actor_lr: float = MID_ACTOR_LR,
                 end_actor_lr: float = END_ACTOR_LR,
                 verbose: int = 1):
        super().__init__(verbose)

                # === Logging setup ===
        self.temp_log_path = os.path.join(SAVE_PATH, "temperature_log.csv")

        # Initialize CSV files with headers if not existing
        if not os.path.exists(self.temp_log_path):
            with open(self.temp_log_path, "w") as f:
                f.write("timestep,temperature\n")

        self.log_std_file = os.path.join(SAVE_PATH, "log_std_log.csv")
        with open(self.log_std_file, "w") as f:
            f.write("step,log_std_mean,log_std_min,log_std_max\n")

        self.warm_start_steps = warm_start_steps
        self.fixed_steps = fixed_steps
        self.total_steps = warm_start_steps + fixed_steps

        self.start_actor_lr = start_actor_lr
        self.mid_actor_lr = mid_actor_lr
        self.end_actor_lr = end_actor_lr

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

    def _on_training_start(self) -> None:
        # Set initial actor.log_std (just once) if available, but keep trainable
        if hasattr(self.model.policy, "actor") and hasattr(self.model.policy.actor, "log_std"):
            actor = self.model.policy.actor
            if isinstance(actor.log_std, torch.nn.Parameter):
                # initialize to a low log_std but let it be trainable afterwards
                with torch.no_grad():
                    actor.log_std.data.fill_(START_LOG_STD)
                if self.verbose:
                    print(f"[ExplorationSchedule] Initialized actor.log_std to {START_LOG_STD} (trainable).")
        else:
            print("[ExplorationSchedule] Model policy has no actor.log_std attribute; skipping initialization.")

        # Freeze actor and ent_coef for warmstart
        self._set_requires_grad(self.model, requires_grad=False)

        # ✅ Keep log_std trainable
        actor = getattr(self.model.policy, "actor", None)
        if actor is not None and hasattr(actor, "log_std"):
            actor.log_std.requires_grad_(True)
            if self.verbose:
                print("[ExplorationSchedule] log_std kept trainable during warmstart.")
        # Keep entropy coefficient trainable
        if hasattr(self.model, "log_ent_coef"):
            self.model.log_ent_coef.requires_grad_(True)
        else:
            print("[ExplorationSchedule] Model has no log_ent_coef attribute; skipping ent_coef freeze.")

    def _on_step(self) -> bool:
        step = self.num_timesteps

        if step <= self.warm_start_steps:
            t = step / max(1, self.warm_start_steps)

        else:
            if not getattr(self, "_unfrozen_actor", False) and step >= self.warm_start_steps:
                self._set_requires_grad(self.model, requires_grad=True)
                actor = getattr(self.model.policy, "actor", None)
                print(f"[ExplorationSchedule] 🔓 Warmstart complete — actor and ent_coef unfrozen.")
                self._unfrozen_actor = True                

        # Logging
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
            print(f"[ExplorationSchedule] step={step} | ent_coef={ent_val}")

            # Log critic LR from the policy's critic optimizer
            try:
                critic_lrs = [g['lr'] for g in self.model.policy.critic.optimizer.param_groups]
                print("Critic LR:", critic_lrs)
            except Exception:
                pass

            # log log_std
            actor = getattr(self.model.policy, "actor", None)
            if actor is not None and hasattr(actor, "log_std"):
                log_std_vals = actor.log_std.detach().cpu().numpy()
                with open(self.log_std_file, "a") as f:
                    f.write(f"{step},{log_std_vals.mean():.6f},{log_std_vals.min():.6f},{log_std_vals.max():.6f}\n")
                # print grad norm
                with torch.no_grad():
                    grad_norm = 0.0
                    for p in actor.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm().item()
                    print(f"[ExplorationCheck] step={step} | total_grad_norm={grad_norm:.6f}")

            # log temperature
            self._log_temperature(step)

        return True

# ---------------- BC-initialized actor wrapper (unchanged except we keep log_std trainable) ----------------
class SACActorWithBC(nn.Module):
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.bc_actor = BCActor(obs_dim).to(device)
        self.bc_actor.eval()
        self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * START_LOG_STD)
        self._eps = 1e-6

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


    def _dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)
        pressures, suctions = self.bc_actor(obs)
        mu = torch.cat([pressures, suctions], dim=-1)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Return action tensor only. Used by predict/acting paths."""
        mu, std = self._dist_params(obs)
        if deterministic:
            return mu
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.clamp(z, -1.0, 1.0)
        return action
    
    def get_action_and_log_prob(self, obs: torch.Tensor):
        """SB3-compatible helper used during training."""
        mu, std = self._dist_params(obs)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
        action = torch.clamp(z, -1.0, 1.0)
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
        # replace actor and create its optimizer so training works
        self.actor = bc_wrapper
        try:
            current_lr = self.lr_schedule(1.0) if callable(self.lr_schedule) else float(self.lr_schedule)
        except Exception:
            current_lr = 3e-4
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=current_lr, **self.optimizer_kwargs)
        print("[BC] ✅ BC-initialized SAC actor loaded and ready.")

# ---------------- VecEnv ----------------
def make_env_fn(rank: int, seed: int = 0, frame_stack: int = 4, gui: bool = False) -> Callable:
    def _init():
        import numpy as np, torch
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        no_links = 5#int(np.random.choice(NO_LINKS_LIST))
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
        start_lr=START_ACTOR_LR,
        mid_lr=MID_ACTOR_LR,
        end_lr=END_ACTOR_LR
    )



    model = SAC(
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
    )

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

    # Load BC actor weights into policy (keeps log_std trainable)
    bc_weights_path = "rl/models/tasq_bc/low_freq_v6/tasq_bc_ep200.pt"
    if os.path.isfile(bc_weights_path):
        model.policy.load_bc_weights(bc_weights_path, device=model.device)
    else:
        print(f"[BC] Skipping: file not found at {bc_weights_path}")

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
        verbose=1,
    )

    callback = CallbackList([reward_logger, progress_callback, checkpoint_callback, schedule_callback])

    # ---------- Learn ----------
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # ---------- Save ----------
    model.save(os.path.join(SAVE_PATH, "tasq_sac_full"))
    torch.save(model.policy.actor.state_dict(), os.path.join(SAVE_PATH, "tasq_sac_actor.pt"))
    print(f"[SAVE] SAC model + actor saved to {SAVE_PATH}")

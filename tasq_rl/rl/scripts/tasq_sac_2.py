# rl/scripts/tasq_sac_seeded.py
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List

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
SAVE_PATH = "rl/models/tasq_sac/low_freqv6/trial3_22_10_25"
os.makedirs(SAVE_PATH, exist_ok=True)

# Where you saved the reward-filled replay buffer (from the script I provided)
REPLAY_BUFFER_PATH = "datasets/replay_buffers/low_freq_v6/replay_buffer1.npz"

# Training / schedule sizes (total = 200_000)
WARM_START_STEPS = 50_000
EXPLORATION_STEPS = 150_000
DECAY_STEPS = 0
TOTAL_SCHEDULE_STEPS = WARM_START_STEPS + EXPLORATION_STEPS + DECAY_STEPS
TOTAL_TIMESTEPS = TOTAL_SCHEDULE_STEPS  # train for exactly the schedule length (tweak as desired)

# Actor LR schedule endpoints (warm -> mid -> decay-end)
START_ACTOR_LR = 1e-8
MID_ACTOR_LR = 3e-5
END_ACTOR_LR = 1e-5

# Target entropy schedule (will be adjusted during decay to favor exploitation)
# We'll pick a reasonable mid target and make the end target *more negative* to reduce entropy target.
ENTROPY_TARGET_MID_SCALE = -1.0  # target = -0.5 * action_dim (your previous choice)
ENTROPY_TARGET_END_FACTOR = 2.0  # end_target = mid_target * ENTROPY_TARGET_END_FACTOR (more negative -> lower entropy desired)

# Other SAC hyperparams
BUFFER_SIZE = 10_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1

# Vectorized env config
NO_LINKS_LIST = [5, 6, 7]
URDF_BY_LINKS = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}

# ---------------- Utility: actor-optimizer lr setter (your helper, unchanged) ----------------
def set_optimizer_lr_for_actor(model, lr):
    """
    Robustly find optimizers that own the policy/actor parameters and set their lr.
    Searches common places SB3 stores optimizers (model, model.policy).
    """
    import torch

    actor_param_ids = {id(p) for p in model.policy.actor.parameters()}
    found = False

    def scan_namespace(namespace, namespace_name="namespace"):
        nonlocal found
        for name, obj in namespace.items():
            if isinstance(obj, torch.optim.Optimizer):
                # check param groups
                for pg in obj.param_groups:
                    params = pg.get("params", [])
                    if any(id(p) in actor_param_ids for p in params):
                        old_lrs = [pg.get("lr") for pg in obj.param_groups]
                        for pg in obj.param_groups:
                            pg["lr"] = lr
                        #print(f"[BC] set lr={lr} for optimizer '{name}' in {namespace_name} (old lrs={old_lrs})")
                        found = True

    scan_namespace(vars(model), "model.__dict__")

    if hasattr(model, "policy"):
        scan_namespace(vars(model.policy), "model.policy.__dict__")
        try:
            scan_namespace({k: v for k, v in vars(model.policy.__class__).items()}, "model.policy.__class__.__dict__")
        except Exception:
            pass

    if not found:
        print("[BC] Warning: no optimizer param_group matched actor params. "
              "Make sure the actor parameters are included in an optimizer on the model.policy (actor_optimizer).")

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

    def _on_training_start(self) -> None:
        # Set initial actor.log_std (just once) if available, but keep trainable
        if hasattr(self.model.policy, "actor") and hasattr(self.model.policy.actor, "log_std"):
            actor = self.model.policy.actor
            if isinstance(actor.log_std, torch.nn.Parameter):
                # initialize to a low log_std but let it be trainable afterwards
                with torch.no_grad():
                    actor.log_std.data.fill_(-2.5)
                if self.verbose:
                    print("[ExplorationSchedule] Initialized actor.log_std to -2.5 (trainable).")

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

        # Force-set actor lr to very small initially for warmstart
        set_optimizer_lr_for_actor(self.model, self.start_actor_lr)
        if self.verbose:
            print(f"[ExplorationSchedule] Set initial actor LR = {self.start_actor_lr:.2e}")

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # compute actor_lr for this step (linear ramp in warmstart; hold; smooth decay)
        if step <= self.warm_start_steps:
            t = step / max(1, self.warm_start_steps)
            actor_lr = self.start_actor_lr #+ t * (self.mid_actor_lr - self.start_actor_lr)
            target_entropy = self.mid_target
        elif step <= self.warm_start_steps + self.fixed_steps:
            actor_lr = self.mid_actor_lr
            target_entropy = self.mid_target
        else:
            decay_step = step - (self.warm_start_steps + self.fixed_steps)
            t = min(1.0, decay_step / max(1, self.decay_steps))
            smooth_t = 1 - np.exp(-5 * t)
            actor_lr = self.mid_actor_lr + smooth_t * (self.end_actor_lr - self.mid_actor_lr)
            # ramp the target entropy toward a more negative value (encourage lower entropy = exploitation)
            target_entropy = self.mid_target + smooth_t * (self.end_target - self.mid_target)

        # Apply actor_lr to optimizer(s) owning actor params
        set_optimizer_lr_for_actor(self.model, float(actor_lr))

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
            print(f"[ExplorationSchedule] step={step} | actor_lr={actor_lr:.2e} | target_entropy={target_entropy:.3f} | ent_coef={ent_val}")

        if step % 1000 == 0:
            actor = getattr(self.model.policy, "actor", None)
            if actor is not None and hasattr(actor, "log_std"):
                log_std_vals = actor.log_std.detach().cpu().numpy()
                with open(self.log_std_file, "a") as f:
                    f.write(f"{step},{log_std_vals.mean():.6f},{log_std_vals.min():.6f},{log_std_vals.max():.6f}\n")

        return True

# ---------------- BC-initialized actor wrapper (unchanged except we keep log_std trainable) ----------------
class SACActorWithBC(nn.Module):
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.bc_actor = BCActor(obs_dim).to(device)
        self.bc_actor.eval()
        # keep log_std as a trainable parameter (initial low)
        self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * -2.5)

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
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)

        pressures, suctions = self.bc_actor(obs)
        mu = torch.cat([pressures, suctions], dim=-1)

        if deterministic:
            action = mu
        else:
            std = torch.exp(self.log_std)  # log_std is trainable
            noise = torch.randn_like(mu) * std
            action = mu + noise

        return torch.clamp(action, -1.0, 1.0)

    def set_training_mode(self, mode: bool):
        if mode:
            self.bc_actor.train()
        else:
            self.bc_actor.eval()

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

    # -------------------------------------------------------------
    # ✅ Gradient clipping for actor updates 
    # -------------------------------------------------------------
    def _update_actor(self, actor_loss: torch.Tensor) -> None:
        self.actor.optimizer.zero_grad()
        actor_loss.backward()

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in self.actor.parameters() if p.grad is not None])
        )
        print(f"[GradClip] Actor grad norm before clip: {total_norm:.3f}")

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()

# ---------------- VecEnv ----------------
def make_env_fn(rank: int, seed: int = 0, frame_stack: int = 4, gui: bool = False) -> Callable:
    def _init():
        import numpy as np, torch
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
    seed = 43
    set_random_seed(seed)
    n_envs = max(1, min(os.cpu_count() or 1, 8))
    env = make_vec_env(n_envs=n_envs, seed=seed, use_subproc=True, frame_stack=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {n_envs} parallel envs on device: {device}")

    # Create SAC with trainable ent_coef (auto) so changing target_entropy has effect
    # note: we intentionally use ent_coef='auto' so the target_entropy adjustments in the callback matter
    model = SAC(
        policy=BCInitializedSACPolicy,
        env=env,
        learning_rate=1e-4,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        ent_coef=0.01,            # auto entropy
        target_entropy=-16.0,
        verbose=1,
        device=device,
        seed=seed,
        policy_kwargs=dict(max_grad_norm=1.0), 
    )


    # Load BC actor weights into policy (keeps log_std trainable)
    bc_weights_path = "rl/models/tasq_bc/low_freq_v6/tasq_bc_ep200.pt"
    if os.path.isfile(bc_weights_path):
        model.policy.load_bc_weights(bc_weights_path, device=model.device)
    else:
        print(f"[BC] Skipping: file not found at {bc_weights_path}")

    # After creating the model, set actor optimizer lr to start very small
    set_optimizer_lr_for_actor(model, START_ACTOR_LR)
    print("[BC] post-optimizer-lr check:")
    for name, opt in model.__dict__.items():
        if isinstance(opt, torch.optim.Optimizer):
            print(f"  {name}: {[pg.get('lr') for pg in opt.param_groups]}")

    # ---------- Seed replay buffer from saved reward-filled buffer ----------
    if os.path.isfile(REPLAY_BUFFER_PATH):
        print(f"[Seed] Loading replay buffer from {REPLAY_BUFFER_PATH} ...")
        added = seed_replay_buffer_from_npz(model, REPLAY_BUFFER_PATH, verbose=True)
        print(f"[Seed] Added {added} transitions from {REPLAY_BUFFER_PATH}")
    else:
        print(f"[Seed] Replay buffer file not found: {REPLAY_BUFFER_PATH}. Skipping seeding.")

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

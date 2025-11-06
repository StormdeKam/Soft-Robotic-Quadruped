# rl/scripts/tasq_sac_seeded.py
import os
import types
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple
import torch.nn.functional as F

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

from rl.models.architecture.actor_no_tanh import BCActor
from rl.utils.callbacks.reward_logger_callback import RewardLoggerCallback
from rl.utils.callbacks.training_progress_callback import TrainingProgressCallback
from rl.scripts.tasq_env import TASQEnv
from rl.scripts.tasq_test import test_model, test_model_video_cv2

# ---------------- Paths / config ----------------
SAVE_PATH = "rl/models/tasq_sac/low_freqv6/scratch/trial1_05_11_25"
os.makedirs(SAVE_PATH, exist_ok=True)

# Training / schedule sizes (total = 200_000)
TIMESTEPS = 100_000

# Actor LR schedule endpoints (warm -> mid -> decay-end)
LR = 3e-4

# Other SAC hyperparams
BUFFER_SIZE = 50_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.97
TRAIN_FREQ = 3
GRADIENT_STEPS = 1
START_LOG_STD = -0.5 

# Vectorized env config
NO_LINKS_LIST = [5, 6, 7]
URDF_BY_LINKS = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}



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

    model = SAC(
        policy=SACPolicy,
        env=env,
        learning_rate=LR, 
        buffer_size=BUFFER_SIZE,
        learning_starts=10_000,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        ent_coef='auto',
        target_entropy=-16.0,
        verbose=1,
        device=device,
        seed=seed
    )

    # ---------------- Callbacks ----------------
    total_timesteps = TIMESTEPS
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

    callback = CallbackList([reward_logger, progress_callback, checkpoint_callback])

    # ---------- Learn ----------
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # ---------- Save ----------
    model.save(os.path.join(SAVE_PATH, "tasq_sac_full"))
    torch.save(model.policy.actor.state_dict(), os.path.join(SAVE_PATH, "tasq_sac_actor.pt"))
    print(f"[SAVE] SAC model + actor saved to {SAVE_PATH}")

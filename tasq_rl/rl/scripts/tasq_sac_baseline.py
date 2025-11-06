import os
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple

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
SAVE_PATH = "rl/models/tasq_sac/low_freqv6/baseline/trial1_05_11_25"
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

class SACWithBC(SAC):
    def _update_actor(self, batch, actor_lr):
        obs = batch.observations
        pi, log_pi = self.policy.actor.action_log_prob(obs)
        q_values = torch.min(
            self.critic(batch.observations, pi),
            self.critic2(batch.observations, pi)
        )
        actor_loss = (self.ent_coef * log_pi - q_values).mean()
        print("actor loss = ", actor_loss)
        # BC regularization
        if hasattr(self.policy.actor, "compute_bc_loss") and self.policy.actor.bc_weight > 0:
            bc_loss = self.policy.actor.compute_bc_loss(obs, pi)
            actor_loss += self.policy.actor.bc_weight * bc_loss
            print("bc loss = ", bc_loss)
            print("actor loss = ", actor_loss)
        self.policy.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.policy.actor.optimizer.step()

# ---------------- BC-initialized actor wrapper (unchanged except we keep log_std trainable) ----------------
class SACActorWithBC(nn.Module):
    def __init__(self, obs_dim, act_dim, device="cpu", bc_weight=0.2, log_std_init=-0.5):
        super().__init__()
        self.device = device
        self.bc_actor = BCActor(obs_dim).to(device)
        self.bc_actor.eval()
        self.bc_weight = bc_weight
        self.sac_actor = BCActor(obs_dim).to(device)
        self.log_std = nn.Parameter(torch.ones(act_dim, device=device) * log_std_init)
    
    def load_from_bc(self, bc_weights_path):
        state_dict = torch.load(bc_weights_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("bc_actor."):
                new_state_dict[k[len("bc_actor."):]] = v
            else:
                new_state_dict[k] = v
        self.bc_actor.load_state_dict(new_state_dict)
        self.sac_actor.load_state_dict(new_state_dict)
        self.bc_actor.eval()

    def set_training_mode(self, mode: bool):
        if mode:
            self.bc_actor.train()
        else:
            self.bc_actor.eval()

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward method for SB3 SAC.
        Returns sampled actions (tanh-squashed).
        """
        pressures, suctions = self.sac_actor(obs)
        mu = torch.cat([pressures, suctions], dim=-1)
        std = self.log_std.exp().expand_as(mu)
        if deterministic:
            return torch.tanh(mu)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        return action

    def action_log_prob(self, obs: torch.Tensor):
        """Return action and log probability (needed by SAC)."""
        pressures, suctions = self.sac_actor(obs)
        mu = torch.cat([pressures, suctions], dim=-1)
        std = self.log_std.exp().expand_as(mu)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def compute_bc_loss(self, obs, actions):
        """MSE toward BC policy."""
        with torch.no_grad():
            bc_actions = self.bc_actor(obs)
        return nn.functional.mse_loss(actions, bc_actions)

# ---------------- Policy wrapper (unchanged) ----------------
class BCInitializedSACPolicy(SACPolicy):
    def __init__(self, *args, bc_weights_path: str = None, bc_device="cpu", **kwargs):
        """
        bc_weights_path: path to pretrained BC actor weights
        bc_device: device to load BC actor
        """
        # Do not assign self.device here — use bc_device for BC wrapper
        self._bc_device = bc_device
        self._bc_weights_path = bc_weights_path
        super().__init__(*args, **kwargs)  # SB3 will call _build() internally

    def _build(self, lr_schedule) -> None:
        """Inject BC actor before parent _build finishes."""
        super()._build(lr_schedule)

        if self._bc_weights_path is not None:
            # Replace actor with BC wrapper
            obs_dim = self.observation_space.shape[0]
            act_dim = self.action_space.shape[0]
            actor = SACActorWithBC(obs_dim=obs_dim, act_dim=act_dim, device=self._bc_device)
            actor.load_from_bc(self._bc_weights_path)

            self.actor = bc_actor
            # Re-create actor optimizer so SAC updates work
            current_lr = lr_schedule(1.0) if callable(lr_schedule) else float(lr_schedule)
            self.actor.optimizer = self.optimizer_class(
                self.actor.parameters(), lr=current_lr, **self.optimizer_kwargs
            )
            print(f"[BC] ✅ BC actor injected and optimizer created. Device={self._bc_device}")

    def _actor_loss(self, obs, actions=None):
        """Override SAC actor loss to include BC regularization."""
        # original SAC actor loss
        pi, log_pi = self.actor.action_log_prob(obs)
        q_values = torch.min(
            self.critic.forward(obs, pi),
            self.critic2.forward(obs, pi)
        )
        actor_loss = (self.ent_coef * log_pi - q_values).mean()
        print("actor_loss = ", actor_loss)
        # BC regularization
        if hasattr(self.actor, "compute_bc_loss") and self.actor.bc_weight > 0:
            # sample a batch from BC dataset if available
            # for simplicity, here we use the same obs
            bc_loss = self.actor.compute_bc_loss(obs, pi)
            actor_loss += self.actor.bc_weight * bc_loss
            print(bc_loss, actor_loss)
        return actor_loss
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

    bc_weights_path = "rl/models/tasq_bc/low_freq_v6/tasq_bc_ep200.pt"

    model = SACWithBC(
        policy=BCInitializedSACPolicy,
        env=env,
        learning_rate=LR, 
        buffer_size=BUFFER_SIZE,
        learning_starts=0,#10_000,
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
        policy_kwargs={
        "bc_weights_path": bc_weights_path,
        "bc_device": device
    }
    )

    initial_ent_coef = 1
    if model.ent_coef == 'auto':
        with torch.no_grad():
            model.log_ent_coef.data.fill_(np.log(initial_ent_coef))

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

import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from rl.scripts.tasq_env import TASQEnv  # Make sure this path is correct
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# --- Configurations ---
MODEL_DIR = "tasq_rl/models/tasq_sac"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "tasq_sac"

TOTAL_TIMESTEPS = 200_0
N_ENVS = 1  # Increase to parallelize (if TASQEnv supports it)

# --- Setup ---
def make_env():
    return TASQEnv(urdf_path="tasq_rl/urdfs/TASQ_n7.urdf", no_links=7, gui=False)

# Wrap in DummyVecEnv
env = DummyVecEnv([make_env])

# Apply frame stacking (e.g., stack last 4 observations)
stacked_env = VecFrameStack(env, n_stack=4)

#env = make_vec_env(make_env, n_envs=N_ENVS)

# --- Load or create model ---
latest_model_path = None
if os.path.exists(MODEL_DIR):
    files = sorted(os.listdir(MODEL_DIR))
    sac_models = [f for f in files if f.startswith(MODEL_NAME) and f.endswith(".zip")]
    if sac_models:
        latest_model = sorted(sac_models)[-1]
        latest_model_path = os.path.join(MODEL_DIR, latest_model)

if latest_model_path:
    print(f"Loading model: {latest_model_path}")
    model = SAC.load(latest_model_path, env=stacked_env)
else:
    print("Creating new SAC model.")
    model = SAC("MlpPolicy", stacked_env, verbose=1)

# --- Setup checkpoint callback ---
checkpoint_callback = CheckpointCallback(
    save_freq=10_000 // N_ENVS,
    save_path=MODEL_DIR,
    name_prefix=MODEL_NAME,
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# --- Train ---
print(f"Training for {TOTAL_TIMESTEPS} steps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

# --- Save final model with timestamp ---
timestamp = time.strftime("%Y%m%d-%H%M%S")
final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{timestamp}.zip")
model.save(final_model_path)
print(f"Model saved: {final_model_path}")

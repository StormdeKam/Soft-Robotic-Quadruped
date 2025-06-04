import os
import glob
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from r2d2_env import R2D2Env

def make_env():
    return R2D2Env(render_mode=None)

def get_latest_model_path(model_dir):
    models = sorted(glob.glob(os.path.join(model_dir, "sac_r2d2_*.zip")))
    return models[-1] if models else None

def main():
    model_dir = "../models"
    log_dir = "../log/sac_r2d2_tensorboard"
    os.makedirs(model_dir, exist_ok=True)

    env = make_vec_env(make_env, n_envs=1)

    # Try loading latest model
    latest_model_path = get_latest_model_path(model_dir)
    if latest_model_path:
        print(f"Loading model from {latest_model_path}")
        model = SAC.load(latest_model_path, env=env, tensorboard_log=log_dir, device="cuda")
    else:
        print("Creating new model")
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cuda")

    # Train model
    model.learn(total_timesteps=1_000)

    # Save model with a unique name (e.g., timestamp or incremental counter)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"sac_r2d2_{timestamp}")
    model.save(model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()


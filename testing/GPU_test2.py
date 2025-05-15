import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Check PyTorch GPU access
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using device:", torch.cuda.get_device_name(0))

# Create env and model
env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
model = SAC("MlpPolicy", env, device="cuda")  # Explicit GPU use
print("Model loaded on device:", model.device)


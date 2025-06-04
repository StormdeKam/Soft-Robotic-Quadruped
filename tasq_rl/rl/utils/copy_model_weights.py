import torch
from stable_baselines3 import SAC
from rl.scripts.tasq_env import TASQEnv

def load_bc_weights_into_sac_actor(sac_model, bc_weights_path, device='cpu'):
    # Load BC weights
    bc_state_dict = torch.load(bc_weights_path, map_location=device)

    # Get SAC actor policy network
    sac_actor = sac_model.policy.actor

    # Load the weights into SAC actor
    sac_actor.load_state_dict(bc_state_dict)
    print("BC weights loaded into SAC actor.")

# --- Usage example ---
env = TASQEnv(urdf_path="tasq_rl/urdfs/TASQ_n7.urdf", no_links=7, gui=False)
model = SAC("MlpPolicy", env, verbose=1)

# Path to your saved BC weights
bc_model_path = "models/tasq_bc/tasq_bc_20250527-123456.pt"

load_bc_weights_into_sac_actor(model, bc_model_path)

# Now you can continue training SAC starting from BC weights:
model.learn(total_timesteps=200_000)

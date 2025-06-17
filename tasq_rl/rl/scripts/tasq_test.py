import os
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from rl.scripts.tasq_simulator import TASQSimulator  # Adjust path if needed

# --- Behavior Cloning model class (same as your training) ---
import torch.nn as nn

class RNNActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # Use final output from sequence
        return self.fc(last_out)


def stack_states(state_buffer, new_state, max_len=4):
    if len(state_buffer) >= max_len:
        state_buffer.pop(0)
    state_buffer.append(new_state)
    while len(state_buffer) < max_len:
        state_buffer.insert(0, new_state)
    return np.stack(state_buffer, axis=0), state_buffer  # shape:


def run_behavior_cloning(model_path, env, device):
    print(f"Loading BC model from {model_path}")
    
    # Get obs_dim and action_dim from saved model or specify here:
    # You must know obs_dim and action_dim from training, or save in a config
    original_obs_dim = 41 # set your original observation dim (from states shape)
    action_dim = 16  # For example, or get from saved data
    N = 4  # stacking frames
    
    obs_dim = original_obs_dim
    seq_len = N
    model = RNNActor(input_dim=obs_dim, hidden_dim=128, output_dim=action_dim).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    state_buffer = []

    suction_states = np.zeros(4)
    pressures = np.zeros((4,3))
    
    steps = 3000
    for step in range(steps):
        # Get observation from simulator
        normalized_pressures = pressures/200 -1
        state = get_env_observation(env, normalized_pressures)

        # Stack observations
        stacked_state, state_buffer = stack_states(state_buffer, state, max_len=N)
        
        # Convert to tensor
        sequence_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)  # shape: (1, seq_len, obs_dim)
        
        with torch.no_grad():
            action = model(sequence_tensor).cpu().numpy().squeeze()
        
        # Convert predicted action to pressures and suction states here, e.g.:
        d_pressures, suction_states = action_to_pressures_and_suction(action)
        pressures += d_pressures
        pressures = np.clip(pressures,0,400) 
        
        env.apply_pressures(pressures)
        env.apply_suction(suction_states)
        env.step()
    
    env.disconnect()
    print("Behavior cloning run complete.")


def run_sac(model_path, env):
    # Wrap env for Stable Baselines3
    def make_env():
        return env
    
    vec_env = DummyVecEnv([make_env])
    stacked_env = VecFrameStack(vec_env, n_stack=4)
    
    print(f"Loading SAC model from {model_path}")
    model = SAC.load(model_path, env=stacked_env)
    
    obs = stacked_env.reset()
    steps = 1000
    for step in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = stacked_env.step(action)
        stacked_env.envs[0].step()
        if dones[0]:
            obs = stacked_env.reset()
    
    stacked_env.close()
    print("SAC run complete.")


def get_env_observation(env, pressures):
    """
    Compose the observation vector to match training input
    (e.g., concatenate base position, orientation, grfs, etc.)
    """
    base_pos = env.get_base_position()          # shape (3,)
    base_quat = env.get_base_orientation()      # shape (4,)
    base_vel = env.get_base_velocity()          # shape (6,)
    suction_states = env.get_suction_state()
    # Ground reaction forces
    grfs = env.get_grfs()                       # shape (12,)

    # Combine all into one observation array
    obs = np.concatenate([
        base_pos,
        base_quat,
        base_vel,
        pressures.flatten(),
        suction_states,
        grfs
    ]).astype(np.float32)
    # Add other observations as needed
    
    return obs


def action_to_pressures_and_suction(action):
    """
    Convert action vector to pressures and suction states:
    - Assuming action_dim = 16 (12 pressures + 4 suction states)
    - Pressures shape: (4 legs, 3 pressures each)
    - Suction: 4 booleans
    
    Adjust as needed based on your action space.
    """
    d_pressures = (action[:12].reshape(4, 3)) * 2*np.pi  # Scale from [-1,1] to [0,400] mBar
    suction_raw = action[12:]
    suction_states = suction_raw > 0.5

    return d_pressures, suction_states


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TASQ with Behavior Cloning or SAC")
    parser.add_argument("--mode", type=str, choices=["bc", "sac"], required=True, help="Run behavior cloning or SAC")
    parser.add_argument("--bc_model_path", type=str, help="Path to the BC PyTorch model (.pt)")
    parser.add_argument("--sac_model_path", type=str, help="Path to the SAC model (.zip)")
    parser.add_argument("--urdf_path", type=str, default="tasq_rl/urdfs/TASQ_n7.urdf", help="Path to TASQ URDF file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    args = parser.parse_args()
    
    # Create environment (no GUI by default)
    env = TASQSimulator(urdf_path="rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
    
    if args.mode == "bc":
        if not args.bc_model_path:
            raise ValueError("Must provide --bc_model_path for behavior cloning mode")
        run_behavior_cloning(args.bc_model_path, env, args.device)
    
    elif args.mode == "sac":
        if not args.sac_model_path:
            raise ValueError("Must provide --sac_model_path for SAC mode")
        run_sac(args.sac_model_path, env)


# Run this file like: 
# python -m rl.scripts.tasq_test --mode bc --bc_model_path path/to/tasq_bc_20230607-152030.pt
# or
# python -m rl.scripts.tasq_test --mode sac --sac_model_path path/to/tasq_sac_20230607-152030.zip



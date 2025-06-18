import os
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from rl.scripts.tasq_simulator import TASQSimulator
import torch.nn as nn


# --- Updated Behavior Cloning Model ---
class RNNActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, pressure_dim=12, suction_dim=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )
        self.pressure_head = nn.Linear(256, pressure_dim)
        self.suction_head = nn.Linear(256, suction_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        shared = self.shared_fc(last_out)
        pressures = self.pressure_head(shared)
        suction_logits = self.suction_head(shared)
        return pressures, suction_logits


def stack_states(state_buffer, new_state, max_len=4):
    if len(state_buffer) >= max_len:
        state_buffer.pop(0)
    state_buffer.append(new_state)
    while len(state_buffer) < max_len:
        state_buffer.insert(0, new_state)
    return np.stack(state_buffer, axis=0), state_buffer


def run_behavior_cloning(model_path, env, device):
    print(f"Loading BC model from {model_path}")
    
    original_obs_dim = 41  # match your training observation shape
    pressure_dim = 12
    suction_dim = 4
    N = 4  # stacking length

    model = RNNActor(input_dim=original_obs_dim, hidden_dim=128,
                     pressure_dim=pressure_dim, suction_dim=suction_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state_buffer = []
    suction_states = np.zeros(4)
    pressures = np.zeros((4, 3))

    steps = 3000
    for step in range(steps):
        normalized_pressures = pressures / 200 - 1
        state = get_env_observation(env, normalized_pressures)

        stacked_state, state_buffer = stack_states(state_buffer, state, max_len=N)
        sequence_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)

        with torch.no_grad():
            pred_pressures, suction_logits = model(sequence_tensor)
            pred_pressures = pred_pressures.cpu().numpy().squeeze()
            suction_probs = torch.sigmoid(suction_logits).cpu().numpy().squeeze()
            suction_states = suction_probs > 0.5

        d_pressures = pred_pressures.reshape(4, 3)
        pressures += d_pressures
        pressures = np.clip(pressures, 0, 400)

        env.apply_pressures(pressures)
        env.apply_suction(suction_states)
        env.step()

    env.disconnect()
    print("Behavior cloning run complete.")


def run_sac(model_path, env):
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
    base_pos = env.get_base_position()
    base_quat = env.get_base_orientation()
    base_vel = env.get_base_velocity()
    suction_states = env.get_suction_state()
    grfs = env.get_grfs()

    obs = np.concatenate([
        base_pos,
        base_quat,
        base_vel,
        pressures.flatten(),
        suction_states,
        grfs
    ]).astype(np.float32)

    return obs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TASQ with Behavior Cloning or SAC")
    parser.add_argument("--mode", type=str, choices=["bc", "sac"], required=True, help="Run behavior cloning or SAC")
    parser.add_argument("--bc_model_path", type=str, help="Path to the BC PyTorch model (.pt)")
    parser.add_argument("--sac_model_path", type=str, help="Path to the SAC model (.zip)")
    parser.add_argument("--urdf_path", type=str, default="tasq_rl/urdfs/TASQ_n7.urdf", help="Path to TASQ URDF file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    args = parser.parse_args()

    env = TASQSimulator(urdf_path=args.urdf_path, gui=True, no_links=7)

    if args.mode == "bc":
        if not args.bc_model_path:
            raise ValueError("Must provide --bc_model_path for behavior cloning mode")
        run_behavior_cloning(args.bc_model_path, env, args.device)

    elif args.mode == "sac":
        if not args.sac_model_path:
            raise ValueError("Must provide --sac_model_path for SAC mode")
        run_sac(args.sac_model_path, env)

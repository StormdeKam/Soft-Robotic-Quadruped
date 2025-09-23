import os
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from rl.scripts.tasq_simulator import TASQSimulator
from rl.models.architecture.actor import BCActor
from rl.scripts.tasq_sac import SACActorWithBC, BCInitializedSACPolicy
import pybullet as p
import zipfile
import tempfile
import torch.nn as nn
import csv

def load_policy_from_checkpoint(zip_path, device):
    with zipfile.ZipFile(zip_path, "r") as archive:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive.extractall(tmpdir)
            policy_path = os.path.join(tmpdir, "policy.pth")
            state_dict = torch.load(policy_path, map_location=device)
    return state_dict
# --- Behavior Cloning model class (Two-headed architecture) ---

# Observed state indices (skipping base_pos and base_vel)
OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base_angular_velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:37   # goal_pos and orientation
]

import zipfile
import pickle

def load_torch_state_from_zip(zip_path, device):
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open("pytorch_variables.pkl", "r") as f:
            state_dict = pickle.load(f)
    return state_dict

def stack_states(state_buffer, new_state, max_len=4):
    if len(state_buffer) >= max_len:
        state_buffer.pop(0)
    state_buffer.append(new_state)
    while len(state_buffer) < max_len:
        state_buffer.insert(0, new_state)
    return np.concatenate(state_buffer, axis=-1), state_buffer

def test_model(model_path, env, device):
    print(f"Loading model from {model_path}")

    original_obs_dim = len(OBSERVED_INDICES)
    action_dim = 16
    N = 4
    obs_dim = original_obs_dim * N

    if model_path.endswith(".zip"):
        print("[INFO] Detected SB3 checkpoint .zip, loading policy.pth only")
        state_dict = load_policy_from_checkpoint(model_path, device)

        if any(k.startswith("bc_actor.") for k in state_dict.keys()):
            print("[INFO] Detected SACActorWithBC weights")
            model = SACActorWithBC(obs_dim=obs_dim, act_dim=action_dim, device=device).to(device)
        else:
            print("[INFO] Detected BCActor weights")
            model = BCActor(obs_dim).to(device)

        model.load_state_dict(state_dict, strict=False)
        model.eval()


        def model_forward(obs_tensor):
            with torch.no_grad():
                out = model(obs_tensor)
                if isinstance(out, tuple):
                    pred_pressures, pred_suctions = out
                    return torch.cat([pred_pressures, pred_suctions], dim=-1)
                else:
                    return out

    else:
        # --- Load plain torch state_dict ---
        state_dict = torch.load(model_path, map_location=device)

        if any(k.startswith("bc_actor.") for k in state_dict.keys()):
            print("[INFO] Detected SACActorWithBC weights")
            model = SACActorWithBC(obs_dim=obs_dim, act_dim=action_dim, device=device).to(device)
        else:
            print("[INFO] Detected BCActor weights")
            model = BCActor(obs_dim).to(device)

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        def model_forward(obs_tensor):
            with torch.no_grad():
                out = model(obs_tensor)
                if isinstance(out, tuple):
                    pred_pressures, pred_suctions = out
                    return torch.cat([pred_pressures, pred_suctions], dim=-1)
                else:
                    return out

    # --- Run rollout ---
    state_buffer = []
    suction_states = np.array([0, 100, 100, 100], dtype=np.float32)
    pressures = np.ones((4, 3)) * 10
    steps = 2100

    control_frequency = 1./24.
    pybullet_frequency = 1./240.
    steps_per_step = int(control_frequency / pybullet_frequency)
    log_states = []
    log_actions = []
    for step in range(steps):
        state = get_env_observation(env)
        stacked_state, state_buffer = stack_states(state_buffer, state, max_len=N)
        log_states.append(stacked_state.flatten().tolist())
        state_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)

        action_tensor = model_forward(state_tensor)
        action = action_tensor.cpu().numpy().squeeze()
        log_actions.append(action.tolist())
        d_pressures, d_suctions = action_to_pressures_and_suction(action)
        
        #for _ in range(steps_per_step):
        pressures += d_pressures #/ steps_per_step
        pressures = np.clip(pressures, 0, 400)
        suction_states += d_suctions #/ steps_per_step
        suction_states = np.clip(suction_states, 0, 100)

        env.apply_pressures(pressures)
        env.apply_suction(suction_states)
        env.step()

    env.disconnect()
    with open("rl/log/tasq_state_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_states)
    with open("rl/log/tasq_action_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_actions)
    print("Testing run complete.")

def get_env_observation(env, obs=None):
    # Use provided obs if available (already stacked)
    if obs is not None:
        full_obs = obs
    else:
        base_pos = np.array(env.get_base_position())
        base_ori = np.array(env.get_base_orientation())
        base_vel = np.array(env.get_base_velocity_local())
        pressures = np.array(env.get_pressure_state()).flatten()
        normalized_pressures = (pressures / 400).flatten()
        suction_states = np.array(env.get_suction_state())
        normalized_suction = suction_states / 100
        grfs = np.array(env.get_grfs_local_foot()) + np.random.normal(6, 6, size=(4,))
        normalized_grfs = grfs / 30
        goal_pos = np.array([0.30, 0.0])
        goal_yaw = np.array([np.sin(np.pi/4), np.sin(np.pi/4)])  # 45 degrees

        full_obs = np.concatenate([
            base_pos,
            base_ori,
            base_vel,
            normalized_pressures,
            normalized_suction,
            normalized_grfs,
            goal_pos,
            goal_yaw
        ]).astype(np.float32)

    obs = full_obs[OBSERVED_INDICES]
    return obs

def action_to_pressures_and_suction(action):
    d_pressures = (action[:12].reshape(4, 3)) * 20
    d_suctions = action[12:16] *20
    return d_pressures, d_suctions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TASQ with specified model")
    parser.add_argument("--model_path", type=str, help="Path to the PyTorch model (.pt)")
    parser.add_argument("--urdf_path", type=str, default="tasq_rl/urdfs/TASQ_n7.urdf", help="Path to TASQ URDF file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    args = parser.parse_args()

    #env = TASQEnv(urdf_path="rl/urdfs/TASQ_n7.urdf", no_links=7, gui=True)

    env = TASQSimulator(urdf_path="rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
    env.reset_simulation()
    if not args.model_path:
        raise ValueError("Must provide --model_path")
    test_model(args.model_path, env, args.device)


# Run this file like: 
# python -m rl.scripts.tasq_test --model_path path/to/tasq_bc_20230607-152030.pt


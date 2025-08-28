import os
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from rl.scripts.tasq_env import TASQEnv
from rl.scripts.tasq_simulator import TASQSimulator  # Adjust path if needed
from rl.models.actor import BCActor
#from rl.scripts.tasq_sac import MultiHeadSAC, MultiHeadSACPolicy, MultiHeadWrappedPolicy, MultiHeadActor
import pybullet as p

# --- Behavior Cloning model class (Two-headed architecture) ---
import torch.nn as nn

# Observed state indices (skipping base_pos and base_vel)
OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base_angular_velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:37   # goal_pos and orientation
]

def stack_states(state_buffer, new_state, max_len=4):
    if len(state_buffer) >= max_len:
        state_buffer.pop(0)
    state_buffer.append(new_state)
    if len(state_buffer) < max_len:
        while len(state_buffer) < max_len:
            state_buffer.insert(0, new_state)
    return np.concatenate(state_buffer, axis=-1), state_buffer

def run_behavior_cloning(model_path, env, device):
    print(f"Loading BC model from {model_path}")

    original_obs_dim = len(OBSERVED_INDICES)
    action_dim = 16
    N = 4
    obs_dim = original_obs_dim * N

    model = BCActor(obs_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state_buffer = []
    suction_states = np.array([0, 100, 100, 100], dtype=np.float32)  # Initial suction states
    pressures = np.ones((4, 3))*10
    steps = 700

    control_frequency = 1./24.
    pybullet_frequency = 1./240.
    steps_per_step = int(control_frequency / pybullet_frequency)

    for step in range(steps):
        state = get_env_observation(env)
        stacked_state, state_buffer = stack_states(state_buffer, state, max_len=N)
        state_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)

        with torch.no_grad():
            pred_pressures, pred_suctions = model(state_tensor)
            action = torch.cat([pred_pressures, pred_suctions], dim=-1).cpu().numpy().squeeze()
        d_pressures, d_suctions = action_to_pressures_and_suction(action)
        for i in range(steps_per_step):
            
            pressures += d_pressures/steps_per_step
            pressures = np.clip(pressures, 0, 400)
            suction_states += d_suctions/steps_per_step
            suction_states = np.clip(suction_states, 0, 100)
            env.apply_pressures(pressures)
            env.apply_suction(suction_states)
            env.step()

    env.disconnect()
    print("Behavior cloning run complete.")

def run_sac(model_path):
    def make_env():
        return TASQEnv(urdf_path="rl/urdfs/TASQ_n7.urdf", no_links=7, gui=True)

    vec_env = DummyVecEnv([make_env])
    stacked_env = VecFrameStack(vec_env, n_stack=4)

    print(f"Loading SAC model from {model_path}")
    model = MultiHeadSAC.load(
    model_path,
    env=stacked_env,
    custom_objects={
        "policy_class": MultiHeadSACPolicy,
        "actor": MultiHeadWrappedPolicy,
        "actor_net": MultiHeadActor,
    }
) 
    print("model.policy:", type(model.policy))
    print("model.policy.actor:", type(model.policy.actor))


    obs = stacked_env.reset()
    steps = 3000
    for step in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        #print("action_sac = ", action)
        obs, rewards, terminated, truncated = stacked_env.step(action)
        done = terminated[0] or truncated[0].get("TimeLimit.truncated", False)
        if done:
            obs = stacked_env.reset()

    stacked_env.close()
    print("SAC run complete.")

def get_env_observation(env):
    # Full observation
    base_pos = np.array(env.get_base_position())
    base_ori = np.array(env.get_base_orientation())
    base_vel = np.array(env.get_base_velocity_local())
    pressures = np.array(env.get_pressures())
    normalized_pressures = (pressures / 200 - 1).flatten()
    suction_states = np.array(env.get_suction_state())
    normalized_suction = suction_states / 50 - 1
    grfs = np.array(env.get_grfs_local_foot())
    normalized_grfs = grfs / 10
    goal_pos = np.array([0.10, 0.0])
    goal_yaw = np.array([np.sin(0), np.sin(0)])

    full_obs = np.concatenate([
        base_pos,
        base_ori,
        base_vel,
        pressures,
        normalized_suction,
        normalized_grfs,
        goal_pos,
        goal_yaw
    ]).astype(np.float32)

    # Only keep observed indices (skip pos & linear velocity)
    obs = full_obs[OBSERVED_INDICES]
    return obs


def action_to_pressures_and_suction(action):
    d_pressures = (action[:12].reshape(4, 3)) * 20 * np.pi
    d_suctions = action[12:16] *20
    return d_pressures, d_suctions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TASQ with Behavior Cloning or SAC")
    parser.add_argument("--mode", type=str, choices=["bc", "sac"], required=True, help="Run behavior cloning or SAC")
    parser.add_argument("--bc_model_path", type=str, help="Path to the BC PyTorch model (.pt)")
    parser.add_argument("--sac_model_path", type=str, help="Path to the SAC model (.zip)")
    parser.add_argument("--urdf_path", type=str, default="tasq_rl/urdfs/TASQ_n7.urdf", help="Path to TASQ URDF file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    args = parser.parse_args()

    #env = TASQEnv(urdf_path="rl/urdfs/TASQ_n7.urdf", no_links=7, gui=True)

    if args.mode == "bc":
        env = TASQSimulator(urdf_path="rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
        env.reset_simulation()
        if not args.bc_model_path:
            raise ValueError("Must provide --bc_model_path for behavior cloning mode")
        run_behavior_cloning(args.bc_model_path, env, args.device)

    elif args.mode == "sac":
        if not args.sac_model_path:
            raise ValueError("Must provide --sac_model_path for SAC mode")
        run_sac(args.sac_model_path)

# Run this file like: 
# python -m rl.scripts.tasq_test --mode bc --bc_model_path path/to/tasq_bc_20230607-152030.pt
# or
# python -m rl.scripts.tasq_test --mode sac --sac_model_path path/to/tasq_sac_20230607-152030.zip

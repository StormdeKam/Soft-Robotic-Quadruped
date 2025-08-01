import os
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from rl.scripts.tasq_env import TASQEnv
from rl.scripts.tasq_simulator import TASQSimulator  # Adjust path if needed
from rl.scripts.tasq_sac import MultiHeadSAC, MultiHeadSACPolicy, MultiHeadWrappedPolicy, MultiHeadActor
import pybullet as p

# --- Behavior Cloning model class (Two-headed architecture) ---
import torch.nn as nn

def rotate_world_to_local(base_pos, base_quat, point):

    # Vector from base to point (world frame)
    vec = np.array([point[0] - base_pos[0], point[1] - base_pos[1], 0.0])

    # Convert quaternion to rotation matrix (3x3)
    rot_matrix = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

    # Inverse rotation: world -> local
    local_vec = rot_matrix.T @ vec

    # Return only x and y in local frame
    return local_vec[0:2]

class TwoHeadBCActor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.pressure_head = nn.Sequential(
            nn.Linear(256, 12),
            nn.Tanh()
        )
        self.suction_head = nn.Linear(256, 4)
           

    def forward(self, x):
        shared = self.shared(x)
        pressures = self.pressure_head(shared)
        suctions = self.suction_head(shared)
        return pressures, suctions

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

    # Dimensions based on your stacking:
    #   full_t (37), full_t-1 (37), partial_t40 (27), partial_t80 (27), partial_t120 (27), partial_t160 (27), partial_t200 (27)
    obs_dim = 37 * 2 + 5 * 27

    model = TwoHeadBCActor(obs_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # History buffer
    history_buffer = []
    max_history_len = 201  # enough to go back to t-200

    # Initialize actuator states
    suction_states = np.zeros(4)
    pressures = np.ones((4, 3)) * 5

    # Helper to extract partial state
    def extract_partial_state(full_obs):
        # slices based on: pos(3), ori(4), vel(6), press(12), suction(4), grfs(4), goal(2), rel_yaw(2)
        return np.concatenate([
            full_obs[0:3],   # base pos
            full_obs[3:7],   # base ori
            full_obs[13:29], # pressures (12) + suction (4)
            full_obs[33:37]  # goal state (4) (pos_x, pos_y, sin(d_yaw), cos(d_yaw))
        ]) 

    # Main rollout
    steps = 250
    for step in range(steps):
        control_frequency = 1./24.
        pybullet_frequency = 1./240.
        steps_per_step = int(control_frequency / pybullet_frequency)

        normalized_pressures = pressures / 200 - 1
        state = get_env_observation(env, normalized_pressures)

        # --- Build history buffer ---
        history_buffer.append(state)
        if len(history_buffer) > max_history_len:
            history_buffer.pop(0)

        # --- Build stacked state ---
        full_t = history_buffer[-1]
        full_tm1 = history_buffer[-2] if len(history_buffer) > 1 else full_t

        partial_t40 = extract_partial_state(history_buffer[-41]) if len(history_buffer) > 40 else extract_partial_state(full_t)
        partial_t80 = extract_partial_state(history_buffer[-81]) if len(history_buffer) > 80 else extract_partial_state(full_t)
        partial_t120 = extract_partial_state(history_buffer[-121]) if len(history_buffer) > 120 else extract_partial_state(full_t)
        partial_t160 = extract_partial_state(history_buffer[-161]) if len(history_buffer) > 160 else extract_partial_state(full_t)
        partial_t200 = extract_partial_state(history_buffer[-201]) if len(history_buffer) > 200 else extract_partial_state(full_t)

        stacked_state = np.concatenate([
            full_t, full_tm1,
            partial_t40, partial_t80, partial_t120, partial_t160, partial_t200
        ]).astype(np.float32)

        # --- Run through network ---
        state_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            pred_pressures, pred_suctions = model(state_tensor)
            action = torch.cat([pred_pressures, pred_suctions], dim=-1).cpu().numpy().squeeze()
        
        # --- Convert action to actuator targets ---
        d_pressures, suction_states = action_to_pressures_and_suction(action)

        # --- Simulate steps ---
        for i in range(steps_per_step):
            pressures += d_pressures / steps_per_step
            pressures = np.clip(pressures, 0, 400)
            env.apply_pressures(pressures)
            env.apply_suction(suction_states)
            env.step()
        print("d = ", d_pressures)

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

def get_env_observation(env, pressures):
    base_pos = np.array(env.get_base_position())
    base_ori = np.array(env.get_base_orientation())
    base_vel = np.array(env.get_base_velocity_local())
    suction_states = np.array(env.get_suction_state())
    grfs = np.array(env.get_grfs_local_foot())
    base_quat = env.get_base_quaternion()
    global_goal_pos = np.array([0.3, 0.0]) #set the global goal pos
    goal_yaw = 0 # set the global goal yaw
    local_goal_pos = rotate_world_to_local(base_pos[0:2], base_quat, global_goal_pos[0:2])
    base_yaw = np.arctan2(base_ori[2], base_ori[3])
    delta_yaw = goal_yaw - base_yaw
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    sin_delta_yaw = np.sin(delta_yaw)
    cos_delta_yaw = np.cos(delta_yaw)
    rel_goal_yaw = [sin_delta_yaw, cos_delta_yaw]

    obs = np.concatenate([
        base_pos,
        base_ori,
        base_vel,
        pressures.flatten(),
        suction_states,
        grfs,
        local_goal_pos,
        rel_goal_yaw
    ]).astype(np.float32)
    return obs

def action_to_pressures_and_suction(action):
    d_pressures = (action[:12].reshape(4, 3)) * 20 * np.pi
    suction_raw = action[12:16]
    suction_sigmoids = 1/(1 + np.exp(-suction_raw))
    suction_states = suction_sigmoids > 0.5
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

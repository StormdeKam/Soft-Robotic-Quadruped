import os
import time
import torch
import numpy as np
import pybullet as p
import csv
from rl.scripts.tasq_simulator import TASQSimulator
from rl.models.architecture.actor import BCActor
from rl.scripts.tasq_sac import SACActorWithBC
import gymnasium as gym


OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base_angular_velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:35,  # phase signal
    35:39   # goal_pos and orientation

]

N = 4
GAIT_LENGTH = 1400

# ------------------------------
# State utilities
# ------------------------------
def rotate_world_to_local(base_pos, base_quat, point):
    vec = np.array([point[0] - base_pos[0], point[1] - base_pos[1], 0.0])
    rot_matrix = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
    local_vec = rot_matrix.T @ vec
    return local_vec[0:2]

class TASQStateBuffer:
    def __init__(self, grf_buffer_len=100, vel_buffer_len=100):
        self.grf_buffer = []
        self.vel_buffer = []
        self.grf_buffer_len = grf_buffer_len
        self.vel_buffer_len = vel_buffer_len
        self.grf_randomizer = np.random.normal(0, 3, size=(4,))
        self.vel_randomizer = np.random.normal(0, 0.01, size=(6,))

    def update_buffers(self, grfs, base_vel):
        self.grf_buffer.append(grfs)
        if len(self.grf_buffer) > self.grf_buffer_len:
            self.grf_buffer.pop(0)
        self.vel_buffer.append(base_vel)
        if len(self.vel_buffer) > self.vel_buffer_len:
            self.vel_buffer.pop(0)

    def get_mean_grf(self):
        if len(self.grf_buffer) == 0:
            return np.zeros(4)
        mean_grfs = np.mean(self.grf_buffer, axis=0) + self.grf_randomizer
        return mean_grfs

    def get_mean_vel(self):
        if len(self.vel_buffer) == 0:
            return np.zeros(6)
        mean_vel = np.mean(self.vel_buffer, axis=0) + self.vel_randomizer
        return mean_vel

def get_env_observation(env, step, state_buffer: TASQStateBuffer):
    base_pos = np.array(env.get_base_position())
    base_ori = np.array(env.get_base_orientation())
    noisy_ori = base_ori * np.random.normal(1, 0.01, size=4)

    base_vel = np.array(env.get_base_velocity_local())
    state_buffer.update_buffers(np.array(env.get_grfs_local_foot()), base_vel)
    mean_vel = state_buffer.get_mean_vel()
    mean_grfs = state_buffer.get_mean_grf()
    normalized_grfs = mean_grfs / 30

    pressures = np.array(env.get_pressure_state()).flatten()
    normalized_pressures = pressures / 400

    suction_states = np.array(env.get_suction_state())
    normalized_suction = suction_states / 100

    raw_phase1 = ((step % GAIT_LENGTH)/GAIT_LENGTH) * 2 * np.pi 
    phase1 = np.array([np.sin(raw_phase1), np.cos(raw_phase1)]) 

    goal_world = np.array([0.010, 0.0])
    goal_local_xy = rotate_world_to_local(base_pos[0:2], base_ori, goal_world[0:2])
    goal_yaw = 0.0
    sin_cos_yaw = np.array([np.sin(goal_yaw), np.cos(goal_yaw)])

    full_obs = np.concatenate([
        base_pos,
        noisy_ori,
        mean_vel,
        normalized_pressures,
        normalized_suction,
        normalized_grfs,
        phase1,
        goal_local_xy,
        sin_cos_yaw
    ]).astype(np.float32)

    return full_obs[OBSERVED_INDICES]

def stack_states(state_buffer_list, new_state, max_len=N):
    if len(state_buffer_list) >= max_len:
        state_buffer_list.pop(0)
    state_buffer_list.append(new_state)
    while len(state_buffer_list) < max_len:
        state_buffer_list.insert(0, new_state)
    return np.concatenate(state_buffer_list, axis=-1), state_buffer_list

def action_to_pressures_and_suction(action):
    d_pressures = (action[:12].reshape(4, 3)) * 400
    d_suctions = action[12:16] * 100
    return d_pressures, d_suctions

# ------------------------------
# Main testing function
# ------------------------------
def test_model(model_path, urdf_path, device="cuda"):
    env = TASQSimulator(urdf_path=urdf_path, gui=False, no_links=7)
    env.reset_simulation()
    print(f"Loading model from {model_path}")

    original_obs_dim = len(OBSERVED_INDICES)
    action_dim = 16
    obs_dim = original_obs_dim * N

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
    state_dict = torch.load(model_path, map_location=device)
    if any(k.startswith("bc_actor.") for k in state_dict.keys()):
        model = SACActorWithBC(obs_dim=obs_dim, act_dim=action_dim, device=device).to(device)
    else:
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

    state_buffer = TASQStateBuffer()
    stacked_state_buffer = []

    suction_states = np.array([0, 100, 100, 100], dtype=np.float32)
    pressures = np.ones((4, 3)) * 10
    steps = GAIT_LENGTH*5

    log_states = []
    log_actions = []

    control_frequency = 240
    control_period = 1.0 / control_frequency

    for step in range(steps):
        step_start = time.time()
        
        obs = get_env_observation(env, step, state_buffer)

        if step % 100 == 0:
            stacked_state, stacked_state_buffer = stack_states(stacked_state_buffer, obs)
            state_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)

            action_tensor = model_forward(state_tensor)
            action = action_tensor.cpu().numpy().squeeze()
            assert action_space.contains(action), "Invalid action"
            log_actions.append(action.tolist())
            d_pressures, d_suctions = action_to_pressures_and_suction(action)

            log_states.append(stacked_state.flatten().tolist())

        pressures += d_pressures / 100
        pressures = np.clip(pressures, 0, 400)
        suction_states += d_suctions / 100
        suction_states = np.clip(suction_states, 0, 100)
        env.apply_pressures(pressures)
        env.apply_suction(suction_states)
        env.step()

        elapsed = time.time() - step_start
        sleep_time = control_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.disconnect()

    os.makedirs("rl/log", exist_ok=True)
    with open("rl/log/tasq_state_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_states)
    with open("rl/log/tasq_action_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_actions)
    print(f"✅ Logged {len(log_states)} stacked states and actions to CSV.")

# ------------------------------
# Run script
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TASQ with specified model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model (.pt)")
    parser.add_argument("--urdf_path", type=str, default="rl/urdfs/TASQ_n7.urdf", help="Path to TASQ URDF")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    args = parser.parse_args()

    test_model(args.model_path, args.urdf_path, args.device)

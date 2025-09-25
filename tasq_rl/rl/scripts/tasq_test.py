import os
import time
import torch
import numpy as np
from stable_baselines3 import SAC
from rl.scripts.tasq_simulator import TASQSimulator
from rl.models.architecture.actor import BCActor
from rl.models.architecture.SAC import SACActorWithBC
import pybullet as p
import torch.nn as nn
import cv2

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
    while len(state_buffer) < max_len:
        state_buffer.insert(0, new_state)
    return np.concatenate(state_buffer, axis=-1), state_buffer

def test_model(model_path, device, n_runs=10, steps=3500, gui=False):
    print(f"[INFO] Loading model from {model_path}")

    original_obs_dim = len(OBSERVED_INDICES)
    action_dim = 16
    N = 4
    obs_dim = original_obs_dim * N

    # Load state_dict
    state_dict = torch.load(model_path, map_location=device)

    # Detect whether it's BCActor or SACActorWithBC
    if any(k.startswith("bc_actor.") for k in state_dict.keys()):
        print("[INFO] Detected SACActorWithBC weights")
        model = SACActorWithBC(obs_dim=obs_dim, act_dim=action_dim, device=device).to(device)
    else:
        print("[INFO] Detected BCActor weights")
        model = BCActor(obs_dim).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Metrics accumulators
    final_x_positions = []
    mse_y_deviations = []
    failures = 0
    
    no_links_list = [5, 6, 7]
    urdf_dict = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}

    for run in range(n_runs):
        print(f"[RUN] Starting test run {run+1}/{n_runs}")
        no_links = np.random.choice(no_links_list)
        urdf_path = urdf_dict[no_links]
        env = TASQSimulator(urdf_path=urdf_path, gui=gui, no_links=no_links)
        env.reset_simulation()

        state_buffer = []
        suction_states = np.array([0, 100, 100, 100], dtype=np.float32)
        pressures = np.ones((4, 3)) * 10

        y_traj = []
        failed = False

        for step in range(steps):
            state = get_env_observation(env)
            stacked_state, state_buffer = stack_states(state_buffer, state, max_len=N)
            state_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)

            with torch.no_grad():
                out = model(state_tensor)
                if isinstance(out, tuple):
                    pred_pressures, pred_suctions = out
                    action = torch.cat([pred_pressures, pred_suctions], dim=-1).cpu().numpy().squeeze()
                else:
                    action = out.cpu().numpy().squeeze()

            d_pressures, d_suctions = action_to_pressures_and_suction(action)

            pressures += d_pressures
            pressures = np.clip(pressures, 0, 400)
            suction_states += d_suctions
            suction_states = np.clip(suction_states, 0, 100)

            env.apply_pressures(pressures)
            env.apply_suction(suction_states)
            env.step()

            # Collect metrics
            base_pos = np.array(env.get_base_position())
            y_traj.append(base_pos[1])
            if base_pos[2] < 0.05:  # failure condition
                failed = True
                break

        final_x_positions.append(base_pos[0])
        mse_y = np.mean(np.square(y_traj))
        mse_y_deviations.append(mse_y)
        if failed:
            failures += 1

        env.disconnect()

    # Aggregate metrics
    avg_x = float(np.mean(final_x_positions))
    avg_mse_y = float(np.mean(mse_y_deviations))
    failure_rate = failures / n_runs

    results = {
        "avg_final_x": avg_x,
        "avg_mse_y": avg_mse_y,
        "failure_rate": failure_rate,
    }
    return results

import cv2

def test_model_video_cv2(model_path, device, steps=3500, video_path="tasq_test.avi", width=960, height=720):
    print(f"[INFO] Running a video test run and saving to {video_path}")

    render_width = 1920   # full HD
    render_height = 1080
    supersample = 2       # render at 2x resolution internally for anti-aliasing

    # Set camera parameters
    cam_target = [0, 0, 0.1]  # focus near robot
    cam_distance = 0.5           # closer zoom
    cam_yaw = 180                 # degrees
    cam_pitch = -1              # degrees
    fov = 60                      # field of view in degrees
    near = 0.01
    far = 1.5
    width = 1280                  # higher resolution
    height = 720

    original_obs_dim = len(OBSERVED_INDICES)
    action_dim = 16
    N = 4
    obs_dim = original_obs_dim * N

    state_dict = torch.load(model_path, map_location=device)

    if any(k.startswith("bc_actor.") for k in state_dict.keys()):
        model = SACActorWithBC(obs_dim=obs_dim, act_dim=action_dim, device=device).to(device)
    else:
        model = BCActor(obs_dim).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Create environment with GUI or DIRECT mode
    env = TASQSimulator(urdf_path="rl/urdfs/TASQ_n7.urdf", gui=False, no_links=7)
    env.reset_simulation()

    # OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 24, (render_width, render_height))

    state_buffer = []
    suction_states = np.array([0, 100, 100, 100], dtype=np.float32)
    pressures = np.ones((4, 3)) * 10

    control_frequency = 1./24.
    pybullet_frequency = 1./240.
    steps_per_step = int(control_frequency / pybullet_frequency)

    for step in range(steps):
        state = get_env_observation(env)
        stacked_state, state_buffer = stack_states(state_buffer, state, max_len=N)
        state_tensor = torch.from_numpy(stacked_state).float().to(device).unsqueeze(0)

        with torch.no_grad():
            out_action = model(state_tensor)
            if isinstance(out_action, tuple):
                pred_pressures, pred_suctions = out_action
                action = torch.cat([pred_pressures, pred_suctions], dim=-1).cpu().numpy().squeeze()
            else:
                action = out_action.cpu().numpy().squeeze()

        d_pressures, d_suctions = action_to_pressures_and_suction(action)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
            distance=cam_distance,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=float(render_width)/render_height,
            nearVal=near,
            farVal=far
        )
        # Capture high-res image
        cam_img = p.getCameraImage(
            width=render_width * supersample,
            height=render_height * supersample,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = cam_img[2][:, :, :3]
        # Downsample to final resolution for smoother edges
        rgb_array = cv2.resize(rgb_array, (render_width, render_height), interpolation=cv2.INTER_AREA)
        rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        out.write(rgb_bgr)
        #time.sleep(control_frequency)  # to simulate real-time
        for _ in range(steps_per_step):
            pressures += d_pressures / steps_per_step
            pressures = np.clip(pressures, 0, 400)
            suction_states += d_suctions / steps_per_step
            suction_states = np.clip(suction_states, 0, 100)

            env.apply_pressures(pressures)
            env.apply_suction(suction_states)
            env.step()

    out.release()
    env.disconnect()
    print(f"[INFO] Video saved at {video_path}")


def get_env_observation(env, obs=None):
    base_pos = np.array(env.get_base_position())
    base_ori = np.array(env.get_base_orientation())
    base_vel = np.array(env.get_base_velocity_local())
    pressures = np.array(env.get_pressure_state()).flatten()
    normalized_pressures = (pressures / 400).flatten()
    suction_states = np.array(env.get_suction_state())
    normalized_suction = suction_states / 100
    grfs = np.array(env.get_grfs_local_foot())+np.random.normal(6, 6, size=(4,))
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
    d_suctions = action[12:16] * 20
    return d_pressures, d_suctions

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TASQ model evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--runs", type=int, default=10, help="Number of test runs")
    parser.add_argument("--gui", action="store_true", help="Enable GUI for debugging (slows evaluation)")
    args = parser.parse_args()
    
    results = test_model(args.model_path, args.device, n_runs=args.runs, gui=args.gui)
    #"""
    print("\n=== Evaluation Results ===")
    print(f"Average Final X Position: {results['avg_final_x']:.3f}")
    print(f"Average Y-Deviation MSE: {results['avg_mse_y']:.6f}")
    print(f"Failure Rate: {results['failure_rate']*100:.2f}%")
    #
    # Run one video
    video_file = os.path.join("/home/storm-de-kam/TASQ/Soft-Robotic-Quadruped/tasq_rl/rl/models/tasq_bc/minimal_viable_V3", "tasq_test.mp4")
    test_model_video_cv2(args.model_path, args.device, steps=1400, video_path=video_file)
    #"""

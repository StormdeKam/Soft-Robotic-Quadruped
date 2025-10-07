import os
import time
import torch
import numpy as np
import cv2
from rl.scripts.tasq_env import TASQEnv
from rl.models.architecture.actor import BCActor
from rl.models.architecture.SAC import SACActorWithBC
import pybullet as p

# Observed state indices (skip base pos/vel if needed)
OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base angular velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:37   # goal pos and orientation
]

def test_model(model_path, device, n_runs=10, steps=3500, gui=False):
    print(f"[INFO] Loading model from {model_path}")

    # Load state_dict and detect actor type
    state_dict = torch.load(model_path, map_location=device)
    original_obs_dim = len(OBSERVED_INDICES)
    action_dim = 16
    frame_stack = 4
    obs_dim = original_obs_dim * frame_stack

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

        # Create TASQEnv (includes frame stacking and noise)
        env = TASQEnv(urdf_path=urdf_path, no_links=no_links, gui=gui, frame_stack=frame_stack)
        obs, _ = env.reset()

        y_traj = []
        failed = False
        control_period = 1/24.

        for step in range(steps):
            step_start = time.time()
            # Only use observed indices for the network
            state_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)

            with torch.no_grad():
                out = model(state_tensor, deterministic=True)
                if isinstance(out, tuple):
                    pred_pressures, pred_suctions = out
                    action = torch.cat([pred_pressures, pred_suctions], dim=-1).cpu().numpy().squeeze()
                else:
                    action = out.cpu().numpy().squeeze()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            base_pos = env.sim.get_base_position()
            y_traj.append(base_pos[1])

            if base_pos[2] < 0.05:  # failure condition
                failed = True
                break

            if gui:
                elapsed = time.time() - step_start
                sleep_time = control_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if terminated or truncated:
                break

        final_x_positions.append(base_pos[0])
        mse_y = np.mean(np.square(y_traj))
        mse_y_deviations.append(mse_y)
        if failed:
            failures += 1

        env.close()

    avg_x = float(np.mean(final_x_positions))
    avg_mse_y = float(np.mean(mse_y_deviations))
    failure_rate = failures / n_runs

    results = {
        "avg_final_x": avg_x,
        "avg_mse_y": avg_mse_y,
        "failure_rate": failure_rate,
    }
    return results


def test_model_video_cv2(model_path, device, steps=3500, video_path="tasq_test.avi", gui=False):
    print(f"[INFO] Running a video test run and saving to {video_path}")

    render_width, render_height = 1920, 1080
    supersample = 2

    # Load model
    state_dict = torch.load(model_path, map_location=device)
    original_obs_dim = len(OBSERVED_INDICES)
    action_dim = 16
    frame_stack = 4
    obs_dim = original_obs_dim * frame_stack

    if any(k.startswith("bc_actor.") for k in state_dict.keys()):
        model = SACActorWithBC(obs_dim=obs_dim, act_dim=action_dim, device=device).to(device)
    else:
        model = BCActor(obs_dim).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    env = TASQEnv(urdf_path="rl/urdfs/TASQ_n7.urdf", no_links=7, gui=gui, frame_stack=frame_stack)
    obs, _ = env.reset()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 24, (render_width, render_height))

    cam_target = [0, 0, 0.1]
    cam_distance = 0.5
    cam_yaw = 180
    cam_pitch = -1
    fov = 60
    near = 0.01
    far = 1.5

    for step in range(steps):
        state_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            out_action = model(state_tensor, deterministic=True)
            if isinstance(out_action, tuple):
                pred_pressures, pred_suctions = out_action
                action = torch.cat([pred_pressures, pred_suctions], dim=-1).cpu().numpy().squeeze()
            else:
                action = out_action.cpu().numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(action)

        # Render camera
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
        cam_img = p.getCameraImage(
            width=render_width*supersample,
            height=render_height*supersample,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = cam_img[2][:,:,:3]
        rgb_array = cv2.resize(rgb_array, (render_width, render_height), interpolation=cv2.INTER_AREA)
        rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        out.write(rgb_bgr)

        if terminated or truncated:
            break

    out.release()
    env.close()
    print(f"[INFO] Video saved at {video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TASQ model evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--runs", type=int, default=10, help="Number of test runs")
    parser.add_argument("--gui", action="store_true", help="Enable GUI for debugging (slows evaluation)")
    args = parser.parse_args()

    results = test_model(args.model_path, args.device, n_runs=args.runs, gui=args.gui)
    print("\n=== Evaluation Results ===")
    print(f"Average Final X Position: {results['avg_final_x']:.3f}")
    print(f"Average Y-Deviation MSE: {results['avg_mse_y']:.6f}")
    print(f"Failure Rate: {results['failure_rate']*100:.2f}%")

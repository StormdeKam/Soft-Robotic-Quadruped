from multiprocessing import Pool, freeze_support, Manager
import numpy as np
import os
import time
from tqdm import tqdm
from functools import partial
import re
import pybullet as p

from rl.scripts.tasq_simulator import TASQSimulator
from rl.gaits.demo_gait_straight_bezier import DemoGaitStraight
demogait = DemoGaitStraight()
demogait.gait()
demo_pressures = demogait.demo_pressures
demo_suction = demogait.demo_suction


# --- Parameters ---
num_episodes = 1 # Number of simulation rollouts
save_path = "rl/datasets/minimal_viable2"  # Path to save the dataset
os.makedirs(save_path, exist_ok=True)
save_interval = 25  # Episodes per save chunk

no_links_list = [5, 6, 7]
urdf_dict = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}

def rotate_world_to_local(base_pos, base_quat, point):
    vec = np.array([point[0] - base_pos[0], point[1] - base_pos[1], 0.0])
    rot_matrix = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
    local_vec = rot_matrix.T @ vec
    return local_vec[0:2]

def run_episode(episode_idx):
    no_links = np.random.choice(no_links_list)
    #gait = np.random.choice([0,1])  # 0 = straight, 1 = turn
    gait = 0
    if gait == 0:
        gait_length = 700
        #mode = np.random.choice([0,1,2,3])  # forward, left, backward, right
        mode = 0
        pressure_callback = partial(demo_pressures, mode=mode)
        suction_callback = partial(demo_suction, mode=mode)
        gait_cycles = np.random.choice([1,2,3,4,5,7,8])
        total_steps = gait_cycles * gait_length
    """
    else:
        gait_length = 21*10
        mode = np.random.choice([0,1])  # clockwise, counterclockwise
        pressure_callback = partial(demo_pressures_rot, mode=mode)
        suction_callback = partial(demo_suction_rot, mode=mode)
        total_steps = 2 * gait_length
    """
    urdf_path = urdf_dict[no_links]
    sim = TASQSimulator(urdf_path=urdf_path, gui=True, no_links=no_links)
    sim.reset_simulation()

    pressures = np.ones((4,3), dtype=np.float32) *10  # Initial pressures
    suction_states = np.array([0,100,100,100], dtype=np.float32) # Initial suction states
    episode_states = []
    episode_actions = []
    quaternion_states = []

    control_frequency = 1./24.
    pybullet_frequency = 1./240.
    steps_per_step = int(control_frequency / pybullet_frequency)

    for step in range(total_steps):
        base_pos = sim.get_base_position()
        base_ori = sim.get_base_orientation()
        base_vel = sim.get_base_velocity_local()
        grfs = sim.get_grfs_local_foot()
        normalized_grfs = grfs / 10
        normalized_pressures = (pressures / 200 - 1).flatten()
        normalized_suction = suction_states/50 -1

        state = np.concatenate([
            base_pos,
            base_ori,
            base_vel,
            normalized_pressures,
            normalized_suction,
            normalized_grfs
        ]).astype(np.float32)

        d_pressures = pressure_callback(step)
        d_suctions = suction_callback(step)
        noisy_d_suctions = d_suctions + np.random.normal(0, 0.01, size=(4,))
        noisy_d_suctions = np.clip(noisy_d_suctions, -20, 20)
        noisy_d_pressures = d_pressures + np.random.normal(0, 0.01, size=(4, 3))
        noisy_d_pressures = np.clip(noisy_d_pressures, -20*np.pi, 20*np.pi)

        action = np.concatenate([(noisy_d_pressures / (20*np.pi)).flatten(), (noisy_d_suctions / 20)]).astype(np.float32)

        for _ in range(steps_per_step):
            pressures += noisy_d_pressures / steps_per_step
            pressures = np.clip(pressures, 0, 400)
            suction_states += noisy_d_suctions / steps_per_step
            suction_states = np.clip(suction_states, 0, 100)
            sim.apply_pressures(pressures)
            sim.apply_suction(suction_states)
            sim.step()
        episode_states.append(state)
        episode_actions.append(action)
        quaternion_states.append(sim.get_base_quaternion())

    # Compute final goal info to augment states
    goal_pos = sim.get_base_position()
    goal_ori = sim.get_base_orientation()
    goal_yaw = np.arctan2(goal_ori[2], goal_ori[3])

    new_episode_states = []
    for i in range(len(episode_states)):
        base_pos = episode_states[i][0:3]
        base_quat = quaternion_states[i]
        base_yaw = np.arctan2(episode_states[i][5], episode_states[i][6])

        local_goal_xy = rotate_world_to_local(base_pos[0:2], base_quat, goal_pos[0:2])
        delta_yaw = goal_yaw - base_yaw
        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
        sin_delta_yaw = np.sin(delta_yaw)
        cos_delta_yaw = np.cos(delta_yaw)

        extended_state = np.concatenate([episode_states[i], local_goal_xy, [sin_delta_yaw, cos_delta_yaw]])
        new_episode_states.append(extended_state)

    sim.disconnect()

    episode_states_np = np.array(new_episode_states, dtype=np.float32)
    episode_actions_np = np.array(episode_actions, dtype=np.float32)
    episode_length = len(episode_states_np)

    return episode_states_np, episode_actions_np, episode_length, episode_idx

def load_sorted_files(folder, prefix):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.npy')]
    def extract_num(f):
        match = re.search(rf"{prefix}_(\d+)\.npy", f)
        return int(match.group(1)) if match else -1
    files_sorted = sorted(files, key=extract_num)
    return [os.path.join(folder, f) for f in files_sorted]

def merge_npy_chunks(folder, prefix, output_name):
    files = load_sorted_files(folder, prefix)
    if not files:
        print(f"No files found with prefix '{prefix}' in {folder}")
        return
    print(f"Merging {len(files)} chunks for '{prefix}'...")
    arrays = [np.load(f) for f in files]
    merged = np.concatenate(arrays, axis=0)
    out_path = os.path.join(folder, output_name)
    np.save(out_path, merged)
    print(f"Saved merged file '{out_path}' with shape {merged.shape}")

if __name__ == "__main__":
    freeze_support()
    manager = Manager()
    lock = manager.Lock()

    args = [(i, lock) for i in range(num_episodes)]

    all_states_buffer = []
    all_actions_buffer = []
    all_episode_lengths_buffer = []

    start_time = time.time()

    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=num_episodes) as pbar:
            for states_np, actions_np, ep_len, (ep_idx, _) in pool.imap_unordered(run_episode, args):
                all_states_buffer.append(states_np)
                all_actions_buffer.append(actions_np)
                all_episode_lengths_buffer.append(ep_len)

                pbar.update(1)

                elapsed = time.time() - start_time
                eps_done = ep_idx + 1
                eps_rem = num_episodes - eps_done
                time_per_ep = elapsed / eps_done
                est_rem_time = eps_rem * time_per_ep
                est_hours = int(est_rem_time // 3600)
                est_minutes = int((est_rem_time // 60) % 60)
                print(f"Episode {eps_done}/{num_episodes} complete — est. time remaining: {est_hours}h {est_minutes}m")

                # Save every save_interval episodes
                if eps_done % save_interval == 0:
                    chunk_id = (eps_done // save_interval) - 1

                    states_concat = np.concatenate(all_states_buffer, axis=0)
                    actions_concat = np.concatenate(all_actions_buffer, axis=0)
                    episode_lengths_concat = np.array(all_episode_lengths_buffer, dtype=np.int32)

                    np.save(os.path.join(save_path, f"demo_states_{chunk_id}.npy"), states_concat)
                    np.save(os.path.join(save_path, f"demo_actions_{chunk_id}.npy"), actions_concat)
                    np.save(os.path.join(save_path, f"demo_episode_lengths_{chunk_id}.npy"), episode_lengths_concat)

                    print(f"Saved chunk {chunk_id} with {states_concat.shape[0]} samples and {episode_lengths_concat.shape[0]} episodes")

                    all_states_buffer.clear()
                    all_actions_buffer.clear()
                    all_episode_lengths_buffer.clear()

    # Save any remaining episodes after all are done
    if len(all_states_buffer) > 0:
        chunk_id = num_episodes // save_interval
        states_concat = np.concatenate(all_states_buffer, axis=0)
        actions_concat = np.concatenate(all_actions_buffer, axis=0)
        episode_lengths_concat = np.array(all_episode_lengths_buffer, dtype=np.int32)

        np.save(os.path.join(save_path, f"demo_states_{chunk_id}.npy"), states_concat)
        np.save(os.path.join(save_path, f"demo_actions_{chunk_id}.npy"), actions_concat)
        np.save(os.path.join(save_path, f"demo_episode_lengths_{chunk_id}.npy"), episode_lengths_concat)

        print(f"Saved final chunk {chunk_id} with {states_concat.shape[0]} samples and {episode_lengths_concat.shape[0]} episodes")

    total_time = time.time() - start_time
    print(f"Completed {num_episodes} episodes in {total_time:.2f}s")

    print("Merging all chunked numpy files...")
    merge_npy_chunks(save_path, "demo_states", "demo_states_merged.npy")
    merge_npy_chunks(save_path, "demo_actions", "demo_actions_merged.npy")

    # For episode lengths, merging means concatenation of 1D arrays
    episode_length_files = load_sorted_files(save_path, "demo_episode_lengths")
    all_lengths = []
    for f in episode_length_files:
        all_lengths.append(np.load(f))
    merged_lengths = np.concatenate(all_lengths, axis=0)
    np.save(os.path.join(save_path, "demo_episode_lengths_merged.npy"), merged_lengths)
    print(f"Saved merged episode lengths with shape {merged_lengths.shape}")

    print("Merging complete!")

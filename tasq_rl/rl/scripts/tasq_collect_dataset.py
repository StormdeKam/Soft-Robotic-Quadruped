from multiprocessing import Pool
import numpy as np
import os
from rl.scripts.tasq_simulator import TASQSimulator
import time
from tqdm import tqdm
from multiprocessing import freeze_support, Manager
import re
from functools import partial
from rl.gaits.demo_gait_straight import demo_pressures, demo_suction
from rl.gaits.demo_gait_rotation import demo_pressures_rot, demo_suction_rot
import pybullet as p

# --- Parameters ---
num_episodes = 10_000 #Number of simulation rollouts
save_path = "rl/datasets/goal_directed"
os.makedirs(save_path, exist_ok=True)

all_states = []
all_actions = []

save_interval = 25

# --- Data Collection Loop ---

no_links_list = [5, 6, 7]
urdf_dict = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}
#startingpoints = [0,800,1300,1800]
startingpressures_dict = {0: 0, 800:210, 1300:230, 1800:250}

def quaternion_to_yaw(quat):
    x, y, z, w = quat
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def rotate_world_to_local(base_pos, base_quat, point):

    # Vector from base to point (world frame)
    vec = np.array([point[0] - base_pos[0], point[1] - base_pos[1], 0.0])

    # Convert quaternion to rotation matrix (3x3)
    rot_matrix = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

    # Inverse rotation: world -> local
    local_vec = rot_matrix.T @ vec

    # Return only x and y in local frame
    return local_vec[0:2]

# Multithreading
def run_episode(episode_idx): 
    no_links = np.random.choice(no_links_list)
    gait = np.random.choice([0,1]) # 0 = straight, 1 = turn
    #gait = 1
    #no_links = 7
    if gait == 0:
        gait_length = 23*10
        mode = np.random.choice([0,1,2,3]) # 0 = forward, 1 = left, 2 = backward, 3 = right
        pressure_callback = partial(demo_pressures, mode=mode)
        suction_callback = partial(demo_suction, mode=mode)
        total_steps = 3*gait_length
    elif gait == 1:
        gait_length = 21*10
        mode = np.random.choice([0,1]) # 0 = clockwise, 1 = counterclockwise
        #mode = 1
        pressure_callback = partial(demo_pressures_rot, mode=mode)
        suction_callback = partial(demo_suction_rot, mode=mode)
        total_steps = 2*gait_length
    
    urdf_path = urdf_dict[no_links]
    sim = TASQSimulator(urdf_path=urdf_path, gui=False, no_links=no_links)
    sim.reset_simulation()

    noisy_d_pressures = np.zeros(12)
    episode_states = []
    episode_actions = []

    # Different starting points:
    #startingpoint = np.random.choice(startingpoints)
    startingpoint = 0
    pressures = np.ones((4, 3)) * startingpressures_dict[startingpoint] + 5
    normalized_pressures = (pressures / 200 - 1).flatten()
    suction_states = suction_callback(startingpoint)
    quaternion_states = []

    for step in range(total_steps):
        control_frequency = 1./24.
        pybullet_frequency = 1./240.
        steps_per_step = int(control_frequency / pybullet_frequency)

        base_pos = sim.get_base_position() # [x,y,z] (global) (3,)
        base_ori = sim.get_base_orientation() # [roll, pitch, sin(yaw), cos(yaw)] (local) (4,)
        base_vel = sim.get_base_velocity_local() # [lin_vel_body, ang_vel_body] (local) (6,)
        grfs = sim.get_grfs_local_foot() # (local foot) (4,)
        normalized_pressures = (pressures / 200 - 1).flatten()

        state = np.concatenate([
            base_pos,
            base_ori,
            base_vel,
            normalized_pressures,
            suction_states,
            grfs
        ]).astype(np.float32)

        d_pressures = pressure_callback(step+startingpoint) #* 0.9
        suction_states = suction_callback(step+startingpoint)

        random_gaussian_noise = np.random.normal(loc=0.0, scale=0.01, size=(4, 3))
        noisy_d_pressures = d_pressures + random_gaussian_noise
        noisy_d_pressures = np.clip(noisy_d_pressures, -20 * np.pi, 20 * np.pi)

        suction_states = suction_states.astype(np.float32)
        normalized_d_pressures = noisy_d_pressures / (20 * np.pi)

        quaternion = sim.get_base_quaternion()
        quaternion_states.append(quaternion)

        action = np.concatenate([normalized_d_pressures.flatten(), suction_states])

        for i in range(steps_per_step):
            pressures += noisy_d_pressures/steps_per_step
            pressures = np.clip(pressures, 0, 400)
            
            sim.apply_pressures(pressures)
            sim.apply_suction(suction_states)
            sim.step()

        episode_states.append(state)
        episode_actions.append(action)
        
        if step == total_steps - 1:
            goal_pos = sim.get_base_position()  # global [x,y,z]
            goal_ori = sim.get_base_orientation()  # [roll, pitch, sin(yaw), cos(yaw)]

            # Extract goal yaw angle from sin(yaw), cos(yaw)
            goal_yaw = np.arctan2(goal_ori[2], goal_ori[3])  # sin_yaw, cos_yaw indices

            new_episode_states = []
            for i in range(len(episode_states)):
                base_pos = episode_states[i][0:3]  # global pos x,y,z
                # base_ori = s[3:7]  # [roll, pitch, sin(yaw), cos(yaw)] if needed

                base_yaw = np.arctan2(episode_states[i][5], episode_states[i][6])  # sin(yaw) = s[5], cos(yaw) = s[6]

                # Compute local goal position (x,y)
                local_goal_xy = rotate_world_to_local(base_pos[0:2], quaternion_states[i], goal_pos[0:2])
                
                # Compute relative yaw goal (goal_yaw - base_yaw) normalized
                delta_yaw = goal_yaw - base_yaw
                delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
                sin_delta_yaw = np.sin(delta_yaw)
                cos_delta_yaw = np.cos(delta_yaw)
                # Append local goal info
                extended_state = np.concatenate([episode_states[i], local_goal_xy, [sin_delta_yaw, cos_delta_yaw]]) # (37,)
                new_episode_states.append(extended_state)

    episode_states = new_episode_states 
    sim.disconnect()

    episode_states_np = np.array(episode_states, dtype=np.float32)
    episode_actions_np = np.array(episode_actions, dtype=np.float32)

    return episode_states_np, episode_actions_np, episode_idx

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
    start_time = time.time()
    #args = list(range(num_episodes))
    #start_time = time.time()

    all_states_buffer = []
    all_actions_buffer = []

    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=num_episodes) as pbar:
            for states_np, actions_np, (ep_idx, _) in pool.imap_unordered(run_episode, args):
                all_states_buffer.append(states_np)
                all_actions_buffer.append(actions_np)
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
                if (ep_idx + 1) % save_interval == 0:
                    chunk_id = (ep_idx + 1) // save_interval - 1
                    states_concat = np.concatenate(all_states_buffer, axis=0)
                    actions_concat = np.concatenate(all_actions_buffer, axis=0)
                    np.save(os.path.join(save_path, f"demo_states_{chunk_id}.npy"), states_concat)
                    np.save(os.path.join(save_path, f"demo_actions_{chunk_id}.npy"), actions_concat)
                    print(f"Saved chunk {chunk_id} with {states_concat.shape[0]} samples")
                    

                    # Clear buffers
                    all_states_buffer.clear()
                    all_actions_buffer.clear()

    # Save remaining data after all episodes are done
    if len(all_states_buffer) > 0:
        chunk_id = (num_episodes // save_interval)
        states_concat = np.concatenate(all_states_buffer, axis=0)
        actions_concat = np.concatenate(all_actions_buffer, axis=0)
        np.save(os.path.join(save_path, f"demo_states_{chunk_id}.npy"), states_concat)
        np.save(os.path.join(save_path, f"demo_actions_{chunk_id}.npy"), actions_concat)
        print(f"Saved final chunk {chunk_id} with {states_concat.shape[0]} samples")

    total_time = time.time() - start_time
    print(f"Completed {num_episodes} episodes in {total_time:.2f}s")

    print("Merging all chunked numpy files...")

    merge_npy_chunks(save_path, "demo_states", "demo_states_merged.npy")
    merge_npy_chunks(save_path, "demo_actions", "demo_actions_merged.npy")

    print("Merging complete!")

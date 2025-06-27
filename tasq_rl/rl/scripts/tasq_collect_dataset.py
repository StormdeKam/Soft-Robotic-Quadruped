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


# --- Parameters ---
gait_length = 23*10
num_episodes = 1 #Number of simulation rollouts
total_steps = 6*gait_length
save_path = "rl/datasets/random_gait_cycle_start"
os.makedirs(save_path, exist_ok=True)

all_states = []
all_actions = []

save_interval = 25

# --- Data Collection Loop ---

no_links_list = [5, 6, 7]
urdf_dict = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}
#startingpoints = [0,800,1300,1800]
startingpressures_dict = {0: 0, 800:210, 1300:230, 1800:250}

# Multithreading
def run_episode(episode_idx): 
    no_links = np.random.choice(no_links_list)
    gait = np.random.choice([1,2])
    if gait == 1:
        mode = np.random.choice([0,1,2,3])
        pressure_callback = partial(demo_pressures, mode=mode)
        suction_callback = partial(demo_suction, mode=mode)
    elif gait == 2:
        mode = np.random.choice([0,1])
        pressure_callback = partial(demo_pressures_rot, mode=mode)
        suction_callback = partial(demo_suction_rot, mode=mode)

    urdf_path = urdf_dict[no_links]
    sim = TASQSimulator(urdf_path=urdf_path, gui=True, no_links=no_links)
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

    for step in range(total_steps):
        base_pos = sim.get_base_position()
        base_quat = sim.get_base_orientation()
        base_vel = sim.get_base_velocity()
        grfs = sim.get_grfs()

        state = np.concatenate([
            base_pos,
            base_quat,
            base_vel,
            normalized_pressures,
            suction_states,
            grfs
        ]).astype(np.float32)

        d_pressures = pressure_callback(step+startingpoint) * 0.9
        suction_states = suction_callback(step+startingpoint)

        random_gaussian_noise = np.random.normal(loc=0.0, scale=0.001, size=(4, 3))
        noisy_d_pressures = d_pressures + random_gaussian_noise
        noisy_d_pressures = np.clip(noisy_d_pressures, -2 * np.pi, 2 * np.pi)

        suction_states = suction_states.astype(np.float32)
        normalized_d_pressures = noisy_d_pressures / (2 * np.pi)

        action = np.concatenate([normalized_d_pressures.flatten(), suction_states])

        pressures += noisy_d_pressures
        pressures = np.clip(pressures, 0, 400)
        normalized_pressures = (pressures / 200 - 1).flatten()

        control_frequency = 1./24.
        pybullet_frequency = 1./240.
        steps_per_step = int(control_frequency / pybullet_frequency)

        for i in range(steps_per_step):
            sim.apply_pressures(pressures)
            sim.apply_suction(suction_states)
            sim.step()

        episode_states.append(state)
        episode_actions.append(action)
        
        if step == total_steps - 1:
            goal_pos = sim.get_base_position()
            goal_quat = sim.get_base_orientation()
            episode_states = [np.concatenate([s, goal_pos, goal_quat]) for s in episode_states]
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
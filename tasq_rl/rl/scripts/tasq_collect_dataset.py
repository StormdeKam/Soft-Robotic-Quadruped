from multiprocessing import Pool
import numpy as np
import os
from rl.scripts.tasq_simulator import TASQSimulator
import time
from tqdm import tqdm
from multiprocessing import freeze_support, Manager
import re

# --- Parameters ---
gait_length = 2300
num_episodes = 1000 #Number of simulation rollouts
total_steps = 3*gait_length
save_path = "rl/datasets/no_steady_states"
os.makedirs(save_path, exist_ok=True)

all_states = []
all_actions = []

save_interval = 25
# --- Gait callbacks ---
n = 100

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
def demo_pressures(step):
    dn = 1/n
    d_pressures = np.zeros((4,3))
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    rel_rel_step = rel_step % n

    if rel_step < n: #Raise body to neutral position #190
        #print("Raising COM 1")
        d_pressures[:,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*190   
        #[190,190,190,  190,190,190,    190,190,190,    190,190,190]
    elif rel_step < 2*n: #Move COM to leg 3 by making leg 3 smaller and adding pressure to chamber 2 of leg 2 and chamber 3 of leg 4
        #print("Moving COM back")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*190
        d_pressures[1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*210
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*190
        d_pressures[3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*210
        #[0,0,0,  400,400,400,    0,0,0,    400,400,400]
    elif rel_step < 3*n: # Lift leg 1 
        #print("Lifting leg 1")
        d_pressures[0,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[350,0,0,  400,400,400,    0,0,0,    400,400,400]
    elif rel_step < 4*n: # Move leg 1 COM 
        #print("move leg 1 and COM")
        d_pressures[0,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[0,1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[0,350,0,  400,400,400,    0,0,0,    400,400,400]
    elif rel_step < 5*n: # Orient leg 1
        #print("orient leg 1 down")
        d_pressures[0,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[0,350,350,  400,400,400,    0,0,0,    400,400,400]
    elif rel_step < 6*n: # Place leg 1 down
        #print("Place leg 1 down")
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350        
        d_pressures[3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150            
        #[0,350,350,  250,250,250,    350,350,350,    250,250,250]
    elif rel_step < 7*n: # Orient leg 1
        #print("orient leg 1 down")
        d_pressures[0,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[0,1:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200
        #[250,150,150,  250,250,250,    350,350,350,    250,250,250]
    elif rel_step < 7*n+1: # Activate suction leg 1, deactivate suction leg 3
        #print("Activate suction leg 1, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))
        #[250,150,150,  250,250,250,    350,350,350,    250,250,250]
    elif rel_step < 8*n: #Raise body to neutral position (210)
        #print("Raising COM 2")
        d_pressures[0,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40
        d_pressures[0,1:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*60
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*140
        d_pressures[3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40
        #[210,210,210,  210,210,210,    210,210,210,    210,210,210]
    elif rel_step < 9*n: #Move COM to leg 1 by making leg 1 smaller and adding pressure to chamber 3 of leg 2 and chamber 2 of leg 4
        #print("Moving COM back")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*210
        d_pressures[1,:2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40
        d_pressures[1,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*140
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*210
        d_pressures[3,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40
        d_pressures[3,1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*140
        d_pressures[3,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40
        #[0,0,0,  250,250,350,    0,0,0,    250,350,250]
    elif rel_step < 10*n: # Lift leg 3
        #print("Lifting leg 3")
        d_pressures[2,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        #[0,0,0,  250,250,350,    150,0,0,    250,350,250]
    elif rel_step < 11*n: # Move leg 3 and COM to front
        d_pressures=np.zeros((4,3))
        d_pressures[1,2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[3,1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[0,0,0,  250,250,250,    150,0,0,    250,250,250]
    elif rel_step < 12*n: # Place leg 3 down
        #print("Place leg 3 down")
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #d_pressures[2,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,1:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[350,350,350,  150,150,150,    150,350,350,    150,150,150]
    elif rel_step < 12*n+1: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))
        #[350,350,350,  150,150,150,    150,350,350,    150,150,150]
    elif rel_step < 13*n: #Raise body to neutral position (230)
        #print("Raising COM")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*120
        d_pressures[1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*80
        d_pressures[2,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*80
        d_pressures[2,1:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*120
        d_pressures[3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*80
        #[230,230,230,  230,230,230,    230,230,230,    230,230,230]
    elif rel_step < 14*n: #Move COM to leg 2 and 3 by making leg 2 and 3 smaller and adding pressure to chamber 2 of leg 1 and chamber 3 of leg 3
        #print("Moving COM back")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*170
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*230
        d_pressures[2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*170
        d_pressures[3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*20
        #[400,400,400,  0,0,0,    400,400,400,    250,250,250]
    elif rel_step < 15*n: # Lift leg 4
        #print("Lifting leg 4")
        d_pressures[3,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[3,1:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[400,400,400,  0,0,0,    400,400,400,    0,350,350]
    elif rel_step < 16*n: # Move leg 4 and COM to front
        #print("move leg 4 to front")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150 
        d_pressures[3,2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[250,250,250,  0,0,0,    400,400,400,    150,350,0]
    elif rel_step < 17*n: # Place leg 4 down
        #print("Place leg 4 down")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[3,1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*300
        d_pressures[3,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        #[150,150,150,  350,350,350,    400,400,400,    150,50,250]
    elif rel_step < 17*n+1: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))
        #[150,150,150,  350,350,350,    400,400,400,    150,50,250]
    elif rel_step < 18*n: #Raise body to neutral position (#250)
        #print("Raising COM 3")
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[3,1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200
        #[250,250,250,  250,250,250,    250,250,250,    250,250,250]
    elif rel_step < 19*n: #Move COM to leg 3 and 4 by making leg 3 and 4 smaller and adding pressure to chamber 1 and 3 of leg 1
        #print("Moving COM back")
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        #[400,400,400,  0,0,0,    400,400,400,    0,0,0]
    elif rel_step < 20*n: # Lift leg 2
        #print("Lifting leg 2")
        d_pressures[1,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[1,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[400,400,400,  250,0,350,    400,400,400,    0,0,0]
    elif rel_step < 21*n: # Move leg 2 and COM to front
        #print("move leg 2 to front")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        #[250,250,250,  250,0,350,    250,250,250,    0,0,0]
    elif rel_step < 22*n: # Place leg 2 down
        #print("Place leg 2 down")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 
        d_pressures[1,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[1,1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[1,2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[150,150,150,  100,350,100,    100,100,100,    350,350,350]
    elif rel_step < 22*n+1: # Activate suction leg 2, deactivate suction leg 3
        #print("Activate suction leg 2, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))

    elif rel_step < 23*n: #Zero pressure position
        #print("Zero pressure position")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[1,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[1,1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[1,2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[0,0,0,  0,0,0,    0,0,0,    0,0,0]
    return d_pressures
""" 
    return d_pressures
"""
def demo_suction(step):
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    
    if rel_step < 7*n:
        ss1 = False
        ss2 = True
        ss3 = True
        ss4 = True
    
    elif rel_step < 12*n:
        ss1 = True
        ss2 = True
        ss3 = False
        ss4 = True

    elif rel_step < 17*n:
        ss1 = True
        ss2 = True
        ss3 = True
        ss4 = False

    elif rel_step < 22*n:
        ss1 = True
        ss2 = False
        ss3 = True
        ss4 = True
    
    else:
        ss1 = False
        ss2 = True
        ss3 = True
        ss4 = True
    
    return np.array([ss1,ss2,ss3,ss4])

# --- Data Collection Loop ---

# Multithreading
def run_episode(episode_idx):
    no_links_list = [5, 6, 7]
    no_links = np.random.choice(no_links_list)
    urdf_dict = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}
    urdf_path = urdf_dict[no_links]
    sim = TASQSimulator(urdf_path=urdf_path, gui=False, no_links=no_links)

    noisy_d_pressures = np.zeros(12)
    suction_states = np.zeros(4)
    pressures = np.zeros((4, 3))
    normalized_pressures = np.zeros(12)
    episode_states = []
    episode_actions = []
    normalized_gait_length = np.zeros(1)

    for step in range(total_steps):
        base_pos = sim.get_base_position()
        base_quat = sim.get_base_orientation()
        base_vel = sim.get_base_velocity()
        grfs = sim.get_grfs()
        time_ = sim.get_time()
        normalized_time = time_ / (5 * gait_length)

        state = np.concatenate([
            base_pos,
            base_quat,
            base_vel,
            normalized_pressures,
            suction_states,
            grfs,
            normalized_time,
            normalized_gait_length
        ]).astype(np.float32)

        d_pressures = demo_pressures(step)
        suction_states = demo_suction(step)
        normalized_gait_length = np.zeros(1)

        random_gaussian_noise = np.random.normal(loc=0.0, scale=0.001, size=(4, 3))
        noisy_d_pressures = d_pressures + random_gaussian_noise
        noisy_d_pressures = np.clip(noisy_d_pressures, -2 * np.pi, 2 * np.pi)

        suction_states = suction_states.astype(np.float32)
        normalized_d_pressures = noisy_d_pressures / (2 * np.pi)

        action = np.concatenate([normalized_d_pressures.flatten(), suction_states, normalized_gait_length])

        pressures += noisy_d_pressures
        pressures = np.clip(pressures, 0, 400)
        normalized_pressures = (pressures / 200 - 1).flatten()

        sim.apply_pressures(pressures)
        sim.apply_suction(suction_states)
        sim.step()

        episode_states.append(state)
        episode_actions.append(action)

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
            for states_np, actions_np, ep_idx in pool.imap_unordered(run_episode, args):
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
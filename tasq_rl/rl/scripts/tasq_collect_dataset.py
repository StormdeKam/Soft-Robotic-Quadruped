from multiprocessing import Pool
import numpy as np
import os
from rl.scripts.tasq_simulator import TASQSimulator
import time
from tqdm import tqdm
from multiprocessing import freeze_support, Manager

# --- Parameters ---
gait_length = 2700
num_episodes = 500  # Number of simulation rollouts
total_steps = 3*gait_length
save_path = "rl/datasets"
os.makedirs(save_path, exist_ok=True)

all_states = []
all_actions = []

# --- Gait callbacks ---
n = 100

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
def demo_pressures(step):
    dn = 1/n
    d_pressures = np.zeros((4,3))
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    rel_rel_step = rel_step % n

    if rel_step < n: #Raise body to neutral position
        #print("Raising COM 1")
        d_pressures[:,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250   
        #[250,250,250,  250,250,250,    250,250,250,    250,250,250]
    elif rel_step < 2*n: #Move COM to leg 3 by making leg 3 smaller and adding pressure to chamber 2 of leg 2 and chamber 3 of leg 4
        #print("Moving COM back")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
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
    elif rel_step < 8*n: # Activate suction leg 1, deactivate suction leg 3
        #print("Activate suction leg 1, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))
        #[250,150,150,  250,250,250,    350,350,350,    250,250,250]
    elif rel_step < 9*n: #Raise body to neutral position
        #print("Raising COM 2")
        d_pressures[0,1:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[250,250,250,  250,250,250,    250,250,250,    250,250,250]
    elif rel_step < 10*n: #Move COM to leg 1 by making leg 1 smaller and adding pressure to chamber 3 of leg 2 and chamber 2 of leg 4
        #print("Moving COM back")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[1,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[3,1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[0,0,0,  250,250,350,    0,0,0,    250,350,250]
    elif rel_step < 11*n: # Lift leg 3
        #print("Lifting leg 3")
        d_pressures[2,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        #[0,0,0,  250,250,350,    150,0,0,    250,350,250]
    elif rel_step < 12*n: # Move leg 3 and COM to front
        d_pressures=np.zeros((4,3))
        d_pressures[1,2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[3,1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[0,0,0,  250,250,250,    150,0,0,    250,250,250]
    elif rel_step < 13*n: # Place leg 3 down
        #print("Place leg 3 down")
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #d_pressures[2,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,1:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[350,350,350,  150,150,150,    150,350,350,    150,150,150]
    elif rel_step < 14*n: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))
        #[350,350,350,  150,150,150,    150,350,350,    150,150,150]
    elif rel_step < 15*n: #Raise body to neutral position
        #print("Raising COM")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,1:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[250,250,250,  250,250,250,    250,250,250,    250,250,250]
    elif rel_step < 16*n: #Move COM to leg 2 and 3 by making leg 2 and 3 smaller and adding pressure to chamber 2 of leg 1 and chamber 3 of leg 3
        #print("Moving COM back")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        #[400,400,400,  0,0,0,    400,400,400,    250,250,250]
    elif rel_step < 17*n: # Lift leg 4
        #print("Lifting leg 4")
        d_pressures[3,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[3,1:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[400,400,400,  0,0,0,    400,400,400,    0,350,350]
    elif rel_step < 18*n: # Move leg 4 and COM to front
        #print("move leg 4 to front")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150 
        d_pressures[3,2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[250,250,250,  0,0,0,    400,400,400,    150,350,0]
    elif rel_step < 19*n: # Place leg 4 down
        #print("Place leg 4 down")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[3,1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*300
        d_pressures[3,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        #[150,150,150,  350,350,350,    400,400,400,    150,50,250]
    elif rel_step < 20*n: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))
        #[150,150,150,  350,350,350,    400,400,400,    150,50,250]
    elif rel_step < 21*n: #Raise body to neutral position
        #print("Raising COM 3")
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[3,1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200
        #[250,250,250,  250,250,250,    250,250,250,    250,250,250]
    elif rel_step < 22*n: #Move COM to leg 3 and 4 by making leg 3 and 4 smaller and adding pressure to chamber 1 and 3 of leg 1
        #print("Moving COM back")
        d_pressures[0,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        #[400,400,400,  0,0,0,    400,400,400,    0,0,0]
    elif rel_step < 23*n: # Lift leg 2
        #print("Lifting leg 2")
        d_pressures[1,0]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[1,2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[400,400,400,  250,0,350,    400,400,400,    0,0,0]
    elif rel_step < 24*n: # Move leg 2 and COM to front
        #print("move leg 2 to front")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        #[250,250,250,  250,0,350,    250,250,250,    0,0,0]
    elif rel_step < 25*n: # Place leg 2 down
        #print("Place leg 2 down")
        d_pressures[0,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 
        d_pressures[1,0]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[1,1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        d_pressures[1,2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[150,150,150,  100,350,100,    100,100,100,    350,350,350]
    elif rel_step < 26*n: # Activate suction leg 2, deactivate suction leg 3
        #print("Activate suction leg 2, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))

    elif rel_step < 27*n: #Zero pressure position
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
    
    if rel_step < 8*n:
        ss1 = False
        ss2 = True
        ss3 = True
        ss4 = True
    
    elif rel_step < 14*n:
        ss1 = True
        ss2 = True
        ss3 = False
        ss4 = True

    elif rel_step < 20*n:
        ss1 = True
        ss2 = True
        ss3 = True
        ss4 = False

    elif rel_step < 26*n:
        ss1 = True
        ss2 = False
        ss3 = True
        ss4 = True
    
    else:
        ss1 = True
        ss2 = True
        ss3 = True
        ss4 = True
    
    return np.array([ss1,ss2,ss3,ss4])

# --- Data Collection Loop ---
"""
for episode in range(num_episodes):
    #sim.disconnect()  # Start clean
    sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=False, no_links=7)
    noisy_d_pressures = np.zeros(12)
    suction_states = np.zeros(4)
    pressures = np.zeros((4,3))

    for step in range(total_steps):
        # Get state
        base_pos = sim.get_base_position()          # shape (3,)
        base_quat = sim.get_base_orientation()      # shape (4,)
        base_vel = sim.get_base_velocity()          # shape (6,)
        grfs = sim.get_grfs()                       # shape (12,)

        # Combine all into one observation array
        state = np.concatenate([
            base_pos,
            base_quat,
            base_vel,
            noisy_d_pressures.flatten(),
            suction_states,
            grfs
        ]).astype(np.float32)

        d_pressures = demo_pressures(step)  # (4, 3)
        suction_states = demo_suction(step)  # (4,)
        
        # Flatten action: 12 pressures + 4 suctions = (16,)
        random_gaussian_noise = np.random.normal(loc=0.0, scale=0.1, size=(4,3))
        noisy_d_pressures = d_pressures + random_gaussian_noise
        suction_states = suction_states.astype(np.float32)
        action = np.concatenate([noisy_d_pressures.flatten(), suction_states])
        
        pressures += noisy_d_pressures
        sim.apply_pressures(pressures)
        sim.apply_suction(suction_states)
        sim.step()
        
        all_states.append(state)
        all_actions.append(action)

    sim.disconnect()
    print(f"Episode {episode + 1}/{num_episodes} complete")

# --- Save Collected Data ---
states_array = np.array(all_states, dtype=np.float32)
actions_array = np.array(all_actions, dtype=np.float32)

np.save(os.path.join(save_path, "demo_states.npy"), states_array)
np.save(os.path.join(save_path, "demo_actions.npy"), actions_array)

print(f"Saved {len(states_array)} state-action pairs to '{save_path}'")
"""
# Multithreading
def run_episode(episode_idx):
    no_links_list = [5,6,7]
    no_links = np.random.choice(no_links_list)
    urdf_dict = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}
    urdf_path = urdf_dict[no_links]
    #urdf_path = "rl/urdfs/TASQ_n7.urdf"
    sim = TASQSimulator(urdf_path=urdf_path, gui=False, no_links=no_links)
    noisy_d_pressures = np.zeros(12)
    suction_states = np.zeros(4)
    pressures = np.zeros((4,3))
    normalized_pressures = np.zeros(12)
    episode_states = []
    episode_actions = []
    for step in range(total_steps):
        # Get state
        base_pos = sim.get_base_position()          # shape (3,)
        base_quat = sim.get_base_orientation()      # shape (4,)
        base_vel = sim.get_base_velocity()          # shape (6,)
        grfs = sim.get_grfs()                       # shape (12,)        

        # Combine all into one observation array
        state = np.concatenate([
            base_pos,
            base_quat,
            base_vel,
            normalized_pressures,
            suction_states,
            grfs
        ]).astype(np.float32)
        #print(normalized_time, ": ", suction_states)

        d_pressures = demo_pressures(step)          # (4, 3)
        suction_states = demo_suction(step)         # (4,)
        
        # Flatten action: 12 pressures + 4 suctions = (16,)
        random_gaussian_noise = np.random.normal(loc=0.0, scale=0.001, size=(4,3))
        noisy_d_pressures = d_pressures + random_gaussian_noise
        noisy_d_pressures = np.clip(noisy_d_pressures, -2*np.pi, 2*np.pi)
        #print(noisy_d_pressures.flatten())
        #print("noisy pressure = ", noisy_d_pressures.flatten())
        #print("pressures = ", d_pressures.flatten())
        #print("noise  ratio = ", np.abs((noisy_d_pressures.flatten() -d_pressures.flatten()) / d_pressures.flatten()))
        suction_states = suction_states.astype(np.float32)
        normalized_d_pressures = noisy_d_pressures/(2*np.pi)
        action = np.concatenate([normalized_d_pressures.flatten(), suction_states])
        
        pressures += noisy_d_pressures
        pressures = np.clip(pressures, 0,400)
        normalized_pressures = (pressures/200 -1).flatten()
        sim.apply_pressures(pressures)
        sim.apply_suction(suction_states)
        sim.step()
        
        episode_states.append(state)
        episode_actions.append(action)

    sim.disconnect()
    #endtime = time.time()
    #runtime = starttime - endtime
    #est_rem_time = (runtime/episode_idx) * (num_episodes - episode_idx) # seconds
    #est_rem_time_hours = est_rem_time // 3600 #hours
    #est_rem_time_minutes = est_rem_time // 60 # minutes
    #print(f"Episode {episode_idx + 1}/{num_episodes} complete") # at {endtime}, estimated remaining time: {est_rem_time_hours}h {est_rem_time_minutes}min ")
    # same code as your for-loop inside here
    return episode_states, episode_actions
"""
if __name__ == "__main__":
    freeze_support()  # Optional but recommended for Windows compatibility

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(run_episode, range(num_episodes))

    # Unpack results
    for ep_states, ep_actions in results:
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)

# --- Save Collected Data ---
states_array = np.array(all_states, dtype=np.float32)
actions_array = np.array(all_actions, dtype=np.float32)

np.save(os.path.join(save_path, "demo_states4.npy"), states_array)
np.save(os.path.join(save_path, "demo_actions4.npy"), actions_array)

print(f"Saved {len(states_array)} state-action pairs to '{save_path}'")
"""
if __name__ == "__main__":
    freeze_support()
    manager = Manager()
    lock = manager.Lock()

    args = [(i, lock) for i in range(num_episodes)]
    start_time = time.time()

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=num_episodes) as pbar:
            for i, result in enumerate(pool.imap_unordered(run_episode, args)):
                results.append(result)
                pbar.update(1)
                elapsed = time.time() - start_time
                eps_done = i + 1
                eps_rem = num_episodes - eps_done
                time_per_ep = elapsed / eps_done
                est_rem_time = eps_rem * time_per_ep
                est_hours = int(est_rem_time // 3600)
                est_minutes = int((est_rem_time // 60) % 60)
                print(f"Episode {eps_done}/{num_episodes} complete — est. time remaining: {est_hours}h {est_minutes}m")

    # Unpack results
    for ep_states, ep_actions in results:
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)

    # Save final data
    states_array = np.array(all_states, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    np.save(os.path.join(save_path, "RNN/demo_states.npy"), states_array)
    np.save(os.path.join(save_path, "RNN/demo_actions.npy"), actions_array)

    print(f"Saved {len(states_array)} state-action pairs to '{save_path}'")
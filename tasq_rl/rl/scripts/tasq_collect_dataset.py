from multiprocessing import Pool
import numpy as np
import os
from rl.scripts.tasq_simulator import TASQSimulator

# --- Parameters ---
gait_length = 2600
num_episodes = 20  # Number of simulation rollouts
total_steps = 3 * gait_length
save_path = "rl/datasets"
os.makedirs(save_path, exist_ok=True)

all_states = []
all_actions = []

# --- Gait callbacks ---
n = 100

# --- Pressure and suction functions (copied from your gait demo) ---
def demo_pressures(step):
    dn = 1/n
    d_pressures = np.zeros((4,3))
    gait_cycle = step // gait_length

    if step < gait_cycle*gait_length+n: #Raise body to neutral position
        #print("Raising COM 1")
        d_pressures[:,:] = dn*250   

    elif step < gait_cycle*gait_length+2*n: #Move COM to leg 3 by making leg 3 smaller and adding pressure to chamber 2 of leg 2 and chamber 3 of leg 4
        #print("Moving COM back")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-dn*250
        d_pressures[1,1]=dn*150
        d_pressures[2,:]=-dn*250
        d_pressures[3,2]=dn*150

    elif step < gait_cycle*gait_length+3*n: # Lift leg 1 
        #print("Lifting leg 1")
        d_pressures=np.zeros((4,3))
        d_pressures[0,1:]=dn*350

    elif step < gait_cycle*gait_length+4*n: # Move leg 1 COM 
        #print("move leg 1 and COM")
        d_pressures=np.zeros((4,3))
        #d_pressures[0,1]=-dn*250
        d_pressures[1,1]=-dn*150
        d_pressures[3,2]=-dn*150

    elif step < gait_cycle*gait_length+5*n: # Place leg 1 down
        #print("Place leg 1 down")
        d_pressures=np.zeros((4,3))
        d_pressures[1,2]=-dn*100
        d_pressures[2,:]=dn*350        
        d_pressures[3,1]=-dn*100            

    elif step < gait_cycle*gait_length+6*n: # Orient leg 1
        #print("orient leg 1 down")
        d_pressures[0,0]=dn*150
        #d_pressures[0,1]=dn*150
        d_pressures[0,1:]=-dn*350

    elif step < gait_cycle*gait_length+7*n: # Activate suction leg 1, deactivate suction leg 3
        #print("Activate suction leg 1, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))

    elif step < gait_cycle*gait_length+8*n: #Raise body to neutral position
        #print("Raising COM 2")
        d_pressures=np.zeros((4,3))
        d_pressures[0,0]=dn*100
        d_pressures[0,1:]=dn*250
        d_pressures[1,2]=dn*100
        d_pressures[2,:]=-dn*100
        d_pressures[3,1]=dn*100

    elif step < gait_cycle*gait_length+9*n: #Move COM to leg 1 by making leg 1 smaller and adding pressure to chamber 3 of leg 2 and chamber 2 of leg 4
        #print("Moving COM back")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-dn*250
        d_pressures[1,2]=dn*100
        d_pressures[2,:]=-dn*250
        d_pressures[3,1]=dn*100

    elif step < gait_cycle*gait_length+10*n: # Lift leg 3
        #print("Lifting leg 3")
        d_pressures=np.zeros((4,3))
        d_pressures[2,0]=dn*350

    elif step < gait_cycle*gait_length+11*n: # Move leg 3 and COM to front
        d_pressures=np.zeros((4,3))
        d_pressures[1,2]=-dn*100
        #d_pressures[2,2]=-dn*300
        d_pressures[3,1]=-dn*100

    elif step < gait_cycle*gait_length+12*n: # Place leg 3 down
        #print("Place leg 3 down")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=dn*350
        d_pressures[1,:]=-dn*100
        d_pressures[2,0]=-dn*350
        d_pressures[2,1:]=dn*350
        d_pressures[2,2]=dn*200
        d_pressures[3,:]=-dn*100
    
    elif step < gait_cycle*gait_length+13*n: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))

    elif step < gait_cycle*gait_length+14*n: #Raise body to neutral position
        #print("Raising COM")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-dn*100
        d_pressures[1,:]=dn*100
        d_pressures[2,0]=dn*250
        d_pressures[2,1:]=-dn*100
        d_pressures[3,:]=dn*100

    elif step < gait_cycle*gait_length+15*n: #Move COM to leg 2 and 3 by making leg 2 and 3 smaller and adding pressure to chamber 2 of leg 1 and chamber 3 of leg 3
        #print("Moving COM back")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:2]=dn*150
        d_pressures[1,:]=-dn*250
        d_pressures[2,:]=-dn*250
        #d_pressures[2,2]=dn*150
        d_pressures[3,:]=-dn*250

    elif step < gait_cycle*gait_length+16*n: # Lift leg 4
        #print("Lifting leg 4")
        d_pressures=np.zeros((4,3))
        d_pressures[3,0]=dn*150
        d_pressures[3,1]=dn*350

    elif step < gait_cycle*gait_length+17*n: # Move leg 4 and COM to front
        #print("move leg 4 to front")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:2]=-dn*150
        d_pressures[2,:]=dn*250
        #d_pressures[3,0]=dn*150

    elif step < gait_cycle*gait_length+18*n: # Place leg 4 down
        #print("Place leg 4 down")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-dn*100
        d_pressures[1,:]=dn*350
        #d_pressures[2,:]=-dn*100   
        d_pressures[3,0]=-dn*150     
        d_pressures[3,1]=-dn*350
        d_pressures[3,2]=dn*250
    
    elif step < gait_cycle*gait_length+19*n: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))

    elif step < gait_cycle*gait_length+20*n: #Raise body to neutral position
        #print("Raising COM 3")

        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=dn*100
        d_pressures[1,:]=-dn*100
        #d_pressures[2,:]=dn*100
        d_pressures[3,:2]=dn*250

    elif step < gait_cycle*gait_length+21*n: #Move COM to leg 3 and 4 by making leg 3 and 4 smaller and adding pressure to chamber 1 and 3 of leg 1
        #print("Moving COM back")
        d_pressures=np.zeros((4,3))
        d_pressures[0,0]=dn*150
        d_pressures[0,2]=dn*150
        d_pressures[1,:]=-dn*250
        d_pressures[2,:]=-dn*250
        d_pressures[3,:]=-dn*250

    elif step < gait_cycle*gait_length+22*n: # Lift leg 2
        #print("Lifting leg 2")
        d_pressures=np.zeros((4,3))
        d_pressures[1,0]=dn*150
        d_pressures[1,2]=dn*350

    elif step < gait_cycle*gait_length+23*n: # Move leg 2 and COM to front
        #print("move leg 2 to front")
        d_pressures=np.zeros((4,3))
        d_pressures[0,0]=-dn*150
        d_pressures[0,2]=-dn*150
        d_pressures[2,:]=dn*250

    elif step < gait_cycle*gait_length+24*n: # Place leg 2 down
        #print("Place leg 2 down")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-dn*100
        d_pressures[1,0]=-dn*150
        d_pressures[1,1]=dn*250
        d_pressures[1,2]=-dn*350
        #d_pressures[2,:]=-dn*100
        d_pressures[3,:]=dn*350
    
    elif step < gait_cycle*gait_length+25*n: # Activate suction leg 2, deactivate suction leg 3
        #print("Activate suction leg 2, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))

    elif step < gait_cycle*gait_length+26*n: #Zero pressure position
        #print("Zero pressure position")
        d_pressures=np.zeros((4,3))
        d_pressures[0,:]=-dn*150
        d_pressures[1,1]=-dn*250
        #d_pressures[2,:]=-dn*150
        d_pressures[3,:]=-dn*350

    return d_pressures

def demo_suction(step):
    gait_cycle = step // gait_length
    if step < gait_cycle * gait_length + 7 * n:
        return np.array([False, True, True, True])
    elif step < gait_cycle * gait_length + 13 * n:
        return np.array([True, True, False, True])
    elif step < gait_cycle * gait_length + 19 * n:
        return np.array([True, True, True, False])
    elif step < gait_cycle * gait_length + 25 * n:
        return np.array([True, False, True, True])
    else:
        return np.array([True, True, True, True])

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
    sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=False, no_links=7)
    noisy_d_pressures = np.zeros(12)
    suction_states = np.zeros(4)
    pressures = np.zeros((4,3))
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
        
        episode_states.append(state)
        episode_actions.append(action)

    sim.disconnect()
    #print(f"Episode {episode + 1}/{num_episodes} complete")
    # same code as your for-loop inside here
    return episode_states, episode_actions

if __name__ == "__main__":
    from multiprocessing import freeze_support
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

np.save(os.path.join(save_path, "demo_states2.npy"), states_array)
np.save(os.path.join(save_path, "demo_actions2.npy"), actions_array)

print(f"Saved {len(states_array)} state-action pairs to '{save_path}'")
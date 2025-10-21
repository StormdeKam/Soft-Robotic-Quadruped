import numpy as np

multiplier = 1  # Increase to make gait faster, decrease to make gait slower
# --- Gait callbacks ---
gait_length = int(1200/multiplier)

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
def demo_pressures(step, mode):
    rel_step = step % gait_length
    d_pressures = np.zeros((4,3))
    #pressures:   
    
    #step 1
    #t=0 [10,10,10,  400,400,400  10,10,10,  400,400,400]
    if rel_step < 100/multiplier:
        d_pressures[0,1] = 70
        d_pressures[0,2] = 290

    #t=100 [10,80,300,  400,400,400,  10,10,10,  400,400,400]
    elif rel_step < 200/multiplier:
        d_pressures[0,0] = 390
        d_pressures[0,1] = 70
        d_pressures[0,2] = -290
        d_pressures[1,:] = -390
        d_pressures[2,:] = 390
        d_pressures[3,:] = -390

    #t=200 [400,150,10,  10,10,10,  400,400,400,  10,10,10]
    elif rel_step < 300/multiplier:
        d_pressures[0,1] = 250
        d_pressures[0,2] = 390      

    #t=300 [400,400,400,  10,10,10,  400,400,400, 10,10,10]
    elif rel_step < 400/multiplier:
        d_pressures[1,1] = 290
        d_pressures[1,2] = 70   

    #t=400 [400,400,400,  10,300,80,  400,400,400,  10,10,10,]
    elif rel_step < 500/multiplier:
        d_pressures[1,0] = 390
        d_pressures[1,1] = -290
        d_pressures[1,2] = 70 
        d_pressures[0,:] = -390
        d_pressures[2,:] = -390
        d_pressures[3,:] = 390

    #t=500 [10,10,10,  400,10,150,  10,10,10,  400,400,400]
    elif rel_step < 600/multiplier:
        d_pressures[1,0] = -390
        d_pressures[1,2] = -140 
        d_pressures[0,:] = 390
        d_pressures[2,:] = 390
        d_pressures[3,:] = -390

    #t=600 [400,400,400  10,10,10,  400,400,400,  10,10,10]
    elif rel_step < 700/multiplier:
        d_pressures[3,0] = 290
        d_pressures[3,2] = 210

    #t=700 [400,400,400  10,10,10,  400,400,400,  300,10,220]
    elif rel_step < 800/multiplier:
        d_pressures[0,:] = -390
        d_pressures[1,:] = 390
        d_pressures[2,:] = -390
        d_pressures[3,0] = -290
        d_pressures[3,1] = 390
        d_pressures[3,2] = -110

    #t=800 [10,10,10  400,400,400,  10,10,10,  10,400,110]
    elif rel_step < 900/multiplier:
        d_pressures[3,0] = 390
        d_pressures[3,2] = 290

    #t=900 [10,10,10  400,400,400,  10,10,10  400,400,400,]
    elif rel_step < 1000/multiplier:
        d_pressures[2,0] = 290
        d_pressures[2,1] = 210

    #t=1000 [10,10,10  400,400,400,  300,220,10,  400,400,400,]
    elif rel_step < 1100/multiplier:
        d_pressures[0,:] = 390
        d_pressures[1,:] = -390
        d_pressures[2,0] = -290
        d_pressures[2,1] = -110
        d_pressures[2,2] = 390
        d_pressures[3,:] = -390

    #t=1100 [400,400,400,  10,10,10,  10,110,400,  10,10,10]
    elif rel_step < 1200/multiplier:
        d_pressures[0,:] = -390
        d_pressures[1,:] = 390
        d_pressures[2,1] = -100
        d_pressures[2,2] = -390
        d_pressures[3,:] = 390

    #t=0 [10,10,10,  400,400,400,  10,10,10, 400,400,400]
    return d_pressures

def demo_suction(step, mode):
    rel_step = step % gait_length
    d_suction = np.array([0,0,0,0])

    #suctions:
    # t=0 [0,100,100,100]
    if rel_step < 100/multiplier:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 200/multiplier:
        d_suction = np.array([0,0,-100,100])
    elif rel_step < 400/multiplier:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 500/multiplier:
        d_suction = np.array([0,100,0,-100])
    elif rel_step < 700/multiplier:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 800/multiplier:
        d_suction = np.array([100,-100,0,0])
    elif rel_step < 1000/multiplier:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 1100/multiplier:
        d_suction = np.array([-100,0,100,0])
    else:
        d_suction = np.array([0,0,0,0])

    return d_suction

import numpy as np
import matplotlib.pyplot as plt

# --- Main function ---
def main():
    out_path = "rl/gaits/demo_gait_straight_low_freq.csv"
    all_data = []
    pressures = np.ones(12) * 10
    pressures[3:6] = 400
    pressures[9:12] = 400
    suctions = np.array([0, 0, 100, 0], dtype=np.float32)

    for step in range(gait_length):
        pressures += demo_pressures(step, mode=0).flatten() / (100.0//multiplier)
        suctions += demo_suction(step, mode=0).flatten() / (100.0/multiplier)
        combined = np.concatenate([pressures, suctions])
        all_data.append(combined)

    all_data = np.array(all_data)
    np.savetxt(out_path, all_data, delimiter=",", fmt="%.4f")
    print(f"✅ Saved {gait_length} steps ({all_data.shape[1]} columns) to {out_path}")

    # --- Plot pressures ---
    plt.figure(figsize=(12, 5))
    for i in range(12):
        plt.plot(all_data[:, i], label=f'Pressure {i+1}')
    plt.title("Pressures over Time")
    plt.xlabel("Step")
    plt.ylabel("Pressure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot suctions ---
    plt.figure(figsize=(8, 4))
    for i in range(4):
        plt.plot(all_data[:, 12 + i], label=f'Suction {i+1}')
    plt.title("Suctions over Time")
    plt.xlabel("Step")
    plt.ylabel("Suction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


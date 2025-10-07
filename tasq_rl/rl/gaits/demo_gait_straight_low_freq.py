import numpy as np

# --- Gait callbacks ---
gait_length = 1400

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
def demo_pressures(step, mode):
    rel_step = step % gait_length
    d_pressures = np.zeros((4,3))
    #pressures:   
    
    #step 1
    #t=0 [10,10,10,  10,10,10,  10,10,10,  10,10,10]
    if rel_step < 100:
        d_pressures[1,:] = 385
        d_pressures[3,:] = 385

    #t=100 [10,10,10,  395,395,395  10,10,10,  395,395,395]
    elif rel_step < 200:
        d_pressures[0,1] = 80
        d_pressures[0,2] = 300

    #t=200 [10,90,310,  395,395,395,  10,10,10,  395,395,395]
    elif rel_step < 300:
        d_pressures[0,0] = 390
        d_pressures[0,1] = 60
        d_pressures[0,2] = -300
        d_pressures[1,:] = -380
        d_pressures[2,:] = 390
        d_pressures[3,:] = -380

    #t=300 [400,150,10,  15,15,15,  400,400,400,  15,15,15]
    elif rel_step < 400:
        d_pressures[0,1] = 250
        d_pressures[0,2] = 390      

    #t=400 [400,400,400,  15,15,15,  400,400,400,  15,15,15]
    elif rel_step < 500:
        d_pressures[1,1] = 300
        d_pressures[1,2] = 80   

    #t=500 [400,400,400,  15,315,95,  400,400,400,  15,15,15]
    elif rel_step < 600:
        d_pressures[1,0] = 375
        d_pressures[1,1] = -295
        d_pressures[1,2] = 60 
        d_pressures[0,:] = -380
        d_pressures[2,:] = -380
        d_pressures[3,:] = 375

    #t=600 [20,20,20  390,20,155,  20,20,20,  390,390,390]
    elif rel_step < 700:
        d_pressures[1,0] = -380
        d_pressures[1,1] = -10
        d_pressures[1,2] = -145 
        d_pressures[0,:] = 377
        d_pressures[2,:] = 377
        d_pressures[3,:] = -380

    #t=700 [397,397,397  10,10,10,  397,397,397,  10,10,10]
    elif rel_step < 800:
        d_pressures[3,0] = 310
        d_pressures[3,2] = 225

    #t=800 [397,397,397  10,10,10,  397,397,397,  320,10,235]
    elif rel_step < 900:
        d_pressures[0,:] = -377
        d_pressures[1,:] = 380
        d_pressures[2,:] = -377
        d_pressures[3,0] = -300
        d_pressures[3,1] = 380
        d_pressures[3,2] = -120 

    #t=900 [20,20,20  390,390,390,  20,20,20,  20,390,105]
    elif rel_step < 1000:
        d_pressures[0,:] = -8
        d_pressures[1,:] = 8
        d_pressures[2,:] = -8
        d_pressures[3,0] = 378
        d_pressures[3,1] = 8
        d_pressures[3,2] = 293

    #t=1000 [12,12,12  398,398,398,  12,12,12,  398,398,398]
    elif rel_step < 1100:
        d_pressures[2,0] = 300
        d_pressures[2,1] = 220

    #t=1100 [12,12,12  398,398,398,  312,232,12,  398,398,398]
    elif rel_step < 1200:
        d_pressures[0,:] = 386
        d_pressures[1,:] = -386
        d_pressures[2,0] = -300
        d_pressures[2,1] = -120
        d_pressures[2,2] = 386
        d_pressures[3,:] = -386

    #t=1200 [398,398,398,  12,12,12,  12,112,398,  12,12,12]
    elif rel_step < 1300:
        d_pressures[0,:] = -384
        d_pressures[1,:] = 378
        d_pressures[2,0] = 2
        d_pressures[2,1] = -98
        d_pressures[2,2] = -384
        d_pressures[3,:] = 378

    #t=1300 [14,14,14,  390,390,390,  14,14,14, 390,390,390]
    elif rel_step < 1400:
        d_pressures[0,:] = -4
        d_pressures[1,:] = -380
        d_pressures[2,:] = -4
        d_pressures[3,:] = -380
    #t=1400/t=0 [10,10,10,  10,10,10,  10,10,10,  10,10,10]

    return d_pressures

def demo_suction(step, mode):
    rel_step = step % gait_length
    d_suction = np.array([0,0,0,0])

    #suctions:
    # t=0 [0,100,100,100]
    if rel_step < 200:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 300:
        d_suction = np.array([100,-100,0,0])
    elif rel_step < 500:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 600:
        d_suction = np.array([0,100,0,-100])
    elif rel_step < 800:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 900:
        d_suction = np.array([0,0,-100,100])
    elif rel_step < 1100:
        d_suction = np.array([0,0,0,0])
    elif rel_step < 1200:
        d_suction = np.array([-100,0,100,0])
    else:
        d_suction = np.array([0,0,0,0])

    return d_suction

# --- Main function ---
def main():
    out_path = "rl/gaits/demo_gait_straight_low_freq.csv"
    all_data = []
    pressures = np.ones(12)*10
    suctions = np.array([0,100,100,100], dtype=np.float32)
    for step in range(gait_length):
        pressures += demo_pressures(step, mode=0).flatten() / 100.0
        suctions += demo_suction(step, mode=0).flatten() / 100.0
        combined = np.concatenate([pressures, suctions])
        all_data.append(combined)

    all_data = np.array(all_data)

    np.savetxt(out_path, all_data, delimiter=",", fmt="%.4f")
    print(f"✅ Saved {gait_length} steps ({all_data.shape[1]} columns) to {out_path}")

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

# --- Gait callbacks ---
n = 10
gait_length = n*23

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
def demo_pressures(step, mode):
    if mode == 0: #forward
        l1,l2,l3,l4 = 0,1,2,3
    elif mode == 1: #left
        l1,l2,l3,l4 = 1,2,3,0
    elif mode == 2: #backward
        l1,l2,l3,l4 = 2,3,0,1
    elif mode == 3: #right
        l1,l2,l3,l4 = 3,0,1,2
        
    dn = 1/n
    d_pressures = np.zeros((4,3))
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    rel_rel_step = rel_step % n
    s = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5

    if rel_step < n: #Raise body to neutral position #190
        #print("Raising COM 1")
        d_pressures[:,:] = s*190 
        #[190,190,190,  190,190,190,    190,190,190,    190,190,190]
    elif rel_step < 2*n: # Lifting COM
        d_pressures[l1,:]=-s*180
        d_pressures[l2,:]=s*200
        d_pressures[l3,:]=-s*180
        d_pressures[l4,:]=s*200
        #[10,10,10,  390,390,390,    10,10,10,    390,390,390]
    elif rel_step < 3*n: # Lift leg 1 and 3
        #print("Lifting leg 1 and 3")
        d_pressures[l1,0]=s*170
        d_pressures[l1,:1]=s*350
        d_pressures[l3,2]=s*350
        #[10,360,360,  390,390,390,    180,10,360,    390,390,390]
    elif rel_step < 4*n: # Move leg 1 and 3
        #print("move leg 1 and 3")
        d_pressures[l3,2]=-s*350
        if rel_step < int(3.5*n):
            d_pressures[l3,0]=-s*170
            d_pressures[l3,1]=s*170
        else:
            d_pressures[l3,0]=s*170
            d_pressures[l3,1]=-s*170

        #[10,160,160,  390,390,390,    10,10,160,    390,390,390]
    """
    elif rel_step < 5*n: # Move COM forward
        #print("Move COM forward")
        d_pressures[l3,:]=s*180
        #[10,110,360,  390,390,390,    190,190,190,    390,390,390]
    elif rel_step < 6*n: # Place leg 1 down
        #print("Place leg 1 down")
        d_pressures[l1,0]=s*350
        d_pressures[l1,1]=-s*100
        d_pressures[l1,2]=-s*350
        d_pressures[l2,:]=-s*140
        #d_pressures[l3,:]=s*350        
        d_pressures[l4,:]=-s*140            
        #[360,10,10,  250,250,250,    190,190,190,    250,250,250]
    elif rel_step < 6*n+1: # Activate suction leg 1, deactivate suction leg 3
        #print("Activate suction leg 1, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))
        #[360,10,10,  250,250,250,    190,190,190,    250,250,250]
    elif rel_step < 7*n: # Orient leg 1
        #print("orient leg 1 down")
        d_pressures[l1,0]=-s*350
        #[10,10,10,  250,250,250,    190,190,190,    250,250,250]
    elif rel_step < 8*n: #Raise body to neutral position (210)
        #print("Raising COM 2")
        d_pressures[l1,:]=-s*200
        d_pressures[l2,:]=-s*40
        d_pressures[l3,:]=s*20
        d_pressures[l4,:]=-s*40
        #[210,210,210,  210,210,210,    210,210,210,    210,210,210]
    elif rel_step < 9*n: #Move COM to leg 1
        #print("Moving COM forward")
        d_pressures[l1,:]=-s*200
        d_pressures[l2,:]=s*180
        d_pressures[l3,:]=-s*200
        d_pressures[l4,:]=s*180
        #[10,10,10,  390,390,390,    10,10,10,    390,390,390]
    elif rel_step < 10*n: # Lift leg 3
        #print("Lifting leg 3")
        d_pressures[l3,1:]=s*350
        #[10,10,10,  390,390,390,    10,360,360,    390,390,390]
    elif rel_step < 11*n: # Move leg 3
        d_pressures[l3,0]=s*350
        d_pressures[l3,2]=-s*350
        #[10,10,10,  390,390,390,    360,360,10,    390,390,390]
    elif rel_step < 12*n: # Place leg 3 down
        #print("Place leg 3 down")
        d_pressures[l1,:]=s*190
        d_pressures[l2,:]=-s*190
        d_pressures[l3,:2]=-s*160
        d_pressures[l4,:]=-s*190
        #[200,200,200,  200,200,200,    200,200,200    200,200,200]
    elif rel_step < 12*n+1: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))
        #[200,200,200,  200,200,200,    200,200,200    200,200,200]
    elif rel_step < 13*n: #Raise body to neutral position (230)
        #print("Raising COM")
        d_pressures[:,:]=s*30
        #[230,230,230,  230,230,230,    230,230,230,    230,230,230]
    elif rel_step < 14*n: #Move COM to leg 2 and 3 by making leg 2 and 3 smaller and adding pressure to chamber 2 of leg 1 and chamber 3 of leg 3
        #print("Moving COM back")
        d_pressures=np.zeros((4,3))
        d_pressures[l1,:]=s*170
        d_pressures[l2,:]=-s*220
        d_pressures[l3,:]=s*170
        d_pressures[l4,:]=s*20
        #[400,400,400,  10,10,10,    400,400,400,    250,250,250]
    elif rel_step < 15*n: # Lift leg 4
        #print("Lifting leg 4")
        d_pressures[l4,0]=-s*240
        d_pressures[l4,1:]=s*110
        #[400,400,400,  10,10,10,    400,400,400,    10,360,360]
    elif rel_step < 16*n: # Move leg 4 and COM to front
        #print("move leg 4 to front")
        d_pressures[l1,:]=-s*150
        d_pressures[l4,0]=s*150 
        d_pressures[l4,2]=-s*350
        #[250,250,250,  10,10,10,    400,400,400,    160,360,10]
    elif rel_step < 17*n: # Place leg 4 down
        #print("Place leg 4 down")
        d_pressures[l1,:]=-s*100
        d_pressures[l2,:]=s*350
        d_pressures[l4,1]=-s*300
        d_pressures[l4,2]=s*250
        #[150,150,150,  360,360,360,    400,400,400,    160,60,260]
    elif rel_step < 17*n+1: # Activate suction leg 3, deactivate suction leg 4
        #print("Activate suction leg 3, deactivate suction leg 4")
        d_pressures=np.zeros((4,3))
        #[150,150,150,  360,360,360,    400,400,400,    160,60,260]
    elif rel_step < 18*n: #Raise body to neutral position (#250)
        #print("Raising COM 3")
        d_pressures[l1,:]=s*100
        d_pressures[l2,:]=-s*110
        d_pressures[l3,:]=-s*150
        d_pressures[l4,0]=s*90
        d_pressures[l4,1]=s*190
        d_pressures[l4,2]=-s*10
        #[250,250,250,  250,250,250,    250,250,250,    250,250,250]
    elif rel_step < 19*n: #Move COM to leg 3 and 4 by making leg 3 and 4 smaller and adding pressure to chamber 1 and 3 of leg 1
        #print("Moving COM back")
        d_pressures[l1,:]=s*150
        d_pressures[l2,:]=-s*240
        d_pressures[l3,:]=s*150
        d_pressures[l4,:]=-s*240
        #[400,400,400,  10,10,10,    400,400,400,    10,10,10,]
    elif rel_step < 20*n: # Lift leg 2
        #print("Lifting leg 2")
        d_pressures[l2,0]=s*250
        d_pressures[l2,2]=s*350
        #[400,400,400,  260,10,360,    400,400,400,    10,10,10,]
    elif rel_step < 21*n: # Move leg 2 and COM to front
        #print("move leg 2 to front")
        d_pressures[l1,:]=-s*150
        d_pressures[l3,:]=-s*150
        #[250,250,250,  260,10,360,    250,250,250,    10,10,10,]
    elif rel_step < 22*n: # Place leg 2 down
        #print("Place leg 2 down")
        d_pressures[l1,:]=-s*50 
        d_pressures[l2,0]=-s*150
        d_pressures[l2,1]=s*350
        d_pressures[l2,2]=-s*250
        d_pressures[l3,:]=-s*100
        d_pressures[l4,:]=s*350
        #[200,200,200,  110,360,110,    150,150,150,    360,360,360]
    elif rel_step < 22*n+1: # Activate suction leg 2, deactivate suction leg 3
        #print("Activate suction leg 2, deactivate suction leg 3")
        d_pressures=np.zeros((4,3))
        #[200,200,200,  110,360,110,    150,150,150,    360,360,360]
    elif rel_step < 23*n: #Zero pressure position
        #print("Zero pressure position")
        d_pressures[l1,:]=-s*200
        d_pressures[l2,0]=-s*110
        d_pressures[l2,1]=-s*360
        d_pressures[l2,2]=-s*110
        d_pressures[l3,:]=-s*150
        d_pressures[l4,:]=-s*360
        #[0,0,0,  0,0,0,    0,0,0,    0,0,0]
"""
    return d_pressures
""" 
    return d_pressures
"""
def demo_suction(step, mode):
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    suction_action = np.array([False,False,False,False])

    if mode == 0:
        l1,l2,l3,l4 = 0,1,2,3
    elif mode == 1:
        l1,l2,l3,l4 = 1,2,3,0
    elif mode == 2:
        l1,l2,l3,l4 = 2,3,0,1
    elif mode == 3:
        l1,l2,l3,l4 = 3,0,1,2
    
    if rel_step < 7*n:
        suction_action[l1] = False
        suction_action[l2] = True
        suction_action[l3] = True
        suction_action[l4] = True
    
    elif rel_step < 12*n:
        suction_action[l1] = True
        suction_action[l2] = True
        suction_action[l3] = False
        suction_action[l4] = True

    elif rel_step < 17*n:
        suction_action[l1] = True
        suction_action[l2] = True
        suction_action[l3] = True
        suction_action[l4] = False

    elif rel_step < 22*n:
        suction_action[l1] = True
        suction_action[l2] = False
        suction_action[l3] = True
        suction_action[l4] = True
    
    else:
        suction_action[l1] = False
        suction_action[l2] = True
        suction_action[l3] = True
        suction_action[l4] = True
    
    return suction_action

# --- Run simulation and collect pressures ---
mode = 0
steps = gait_length
pressures_over_time = []
p = np.zeros((4,3))

for step in range(steps):
    p += demo_pressures(step, mode)
    pressures_over_time.append(p.flatten())  # shape (12,)

pressures_over_time = np.array(pressures_over_time)  # shape (steps, 12)

# --- Plotting ---
plt.figure(figsize=(12, 8))
labels = [f"L{leg+1}_C{ch+1}" for leg in range(4) for ch in range(3)]

for i in range(12):
    plt.plot(pressures_over_time[:, i], label=labels[i])

plt.title("Pressure Values over Gait Cycle")
plt.xlabel("Step")
plt.ylabel("Pressure")
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

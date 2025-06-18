from rl.scripts.tasq_simulator import TASQSimulator
import numpy as np
import time

gait_length = 2300
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


sim = TASQSimulator("rl/urdfs/TASQ_n5.urdf", gui=True, no_links=5)
sim.run(steps=3*gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction)
print(sim.get_base_position())
sim.disconnect()

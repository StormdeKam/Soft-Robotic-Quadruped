from rl.scripts.tasq_simulator import TASQSimulator
import numpy as np
import time

gait_length = 2600
n = 100

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
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
""" 
    return d_pressures
"""
def demo_suction(step):
    gait_cycle = step // gait_length
    
    if step < gait_cycle*gait_length+7*n:
        ss1 = False
        ss2 = True
        ss3 = True
        ss4 = True
    
    elif step < gait_cycle*gait_length+13*n:
        ss1 = True
        ss2 = True
        ss3 = False
        ss4 = True

    elif step < gait_cycle*gait_length+19*n:
        ss1 = True
        ss2 = True
        ss3 = True
        ss4 = False

    elif step < gait_cycle*gait_length+25*n:
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

sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
sim.run(steps=3*gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction)
print(sim.get_base_position_and_orientation())
sim.disconnect()

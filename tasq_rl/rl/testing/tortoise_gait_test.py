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
    rel_step = step % gait_length
    rel_rel_step = rel_step % n

    if rel_step < n: #Raise body to neutral position
        #print("Raising body to neutral position")
        d_pressures[:,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250   

    elif rel_step < 2*n: #Tilt body
        #print("Tilting body")
        d_pressures[:2,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #350
        d_pressures[2:,:] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #200

    elif rel_step < 3*n: # COM shift
        #print("Shifting COM to side")
        d_pressures[1,:] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #250
        d_pressures[2,:] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #100


    elif rel_step < 4*n: # Leg 1 step
        #print("Lifting leg 1")
        d_pressures[0,:2] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250 #100

    elif rel_step < 5*n: # Leg 1 step
        #print("Moving leg 1")
        d_pressures[0,0] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #50
        d_pressures[0,1] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200 #300
        d_pressures[0,2] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #300

    elif rel_step < 6*n: # Leg 1 step
        #print("Moving leg 1")
        d_pressures[0,0] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #100
        d_pressures[0,1] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #350
        d_pressures[0,2] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200 #100

    elif rel_step < 7*n: # Leg 1 step
        #print("Moving COM forward and placing leg 1 down")
        d_pressures[2,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150 #250
        d_pressures[3,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #250

    elif rel_step < 8*n: # Leg 1 step
        #print("Placing leg 1 down")
        d_pressures[0,0] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #200
        d_pressures[0,1] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250 #100
        d_pressures[0,2] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #200

    elif rel_step < 9*n: # Tilt body
        #print("Tilting body")
        d_pressures[0,0] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150 #50
        d_pressures[0,1] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #50
        d_pressures[0,2] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150 #50
        d_pressures[1,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50 #300
        d_pressures[3,:] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200 #50

    elif rel_step < 10*n: # Leg 3 step
        #print("Lifting leg 3")
        d_pressures[2,0] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200 #50
        #d_pressures[2,1] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #250
        d_pressures[2,2] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #350

    elif rel_step < 11*n: # Leg 3 step
        #print("Moving leg 3")
        d_pressures[2,0] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200 #250
        d_pressures[2,1] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*300 #50
        d_pressures[2,2] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #250

    elif rel_step < 12*n: # Leg 3 step
        #print("Place leg 3 down")
        d_pressures[2,0] = -(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200 #50
        #d_pressures[2,1] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*300 #50
        d_pressures[2,2] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 #350


    return d_pressures
"""
"""
def demo_suction(step):
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    
    if rel_step < 8*n:
        ss1 = False
        ss2 = True
        ss3 = False
        ss4 = True
    
    elif rel_step < 13*n:
        ss1 = True
        ss2 = True
        ss3 = False
        ss4 = True

    elif rel_step < 19*n:
        ss1 = True
        ss2 = True
        ss3 = True
        ss4 = False

    elif rel_step < 25*n:
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

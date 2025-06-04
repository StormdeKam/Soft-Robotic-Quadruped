from rl.scripts.tasq_simulator import TASQSimulator
import numpy as np
import time

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
def demo_pressures(step):
    n = 200
    dn = 1/n
    
    if step < n: #Raise body 
        #print("Raising COM")
        dp11 = 0
        dp12 = 0
        dp13 = 0
        dp21 = dn*250
        dp22 = dn*250
        dp23 = dn*250
        dp31 = 0
        dp32 = 0
        dp33 = 0
        dp41 = dn*250
        dp42 = dn*250
        dp43 = dn*250

    else:
        dp11 = 0
        dp12 = 0
        dp13 = 0
        dp21 = 0
        dp22 = 0
        dp23 = 0
        dp31 = 0
        dp32 = 0
        dp33 = 0
        dp41 = 0
        dp42 = 0
        dp43 = 0
    

    #"""    
    return np.array([
            [dp11, dp12, dp13],  # Leg 1
            [dp21, dp22, dp23],  # Leg 2
            [dp31, dp32, dp33],  # Leg 3
            [dp41, dp42, dp43]   # Leg 4
        ])
    #"""
    return np.array([
            [dp11, dp12, dp13],  # Leg 1
            [0,0,0],  # Leg 2
            [0,0,0],  # Leg 3
            [0,0,0]   # Leg 4
        ])

def demo_suction(step):
    n = 200
    
    if step < 100*n:
        ss1 = False
        ss2 = True
        ss3 = True
        ss4 = True
    
    else:
        ss1 = False
        ss2 = False
        ss3 = False
        ss4 = False
    
    return np.array([ss1,ss2,ss3,ss4])

sim = TASQSimulator("tasq_rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
sim.run(steps=200000, pressure_callback=demo_pressures, suction_callback=demo_suction)
print(sim.get_base_position())
sim.disconnect()

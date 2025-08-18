import numpy as np

# --- Gait callbacks ---
n = 10
gait_length = n*21

# Dummy callback that sends oscillating pressures for 4 legs (3 per leg)
def demo_pressures_rot(step, mode):
    l1,l2,l3,l4 = 0,1,2,3
    if mode == 0: #turn clockwise
        p1,p2,p3 = 0,1,2
    elif mode == 1: #turn counterclockwise
        p1,p2,p3 = 0,2,1

        
    dn = 1/n
    d_pressures = np.zeros((4,3))
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    rel_rel_step = rel_step % n

    if rel_step < n: #Raise body to neutral position #190
        d_pressures[:,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*190   
        #[190,190,190,  190,190,190,    190,190,190,    190,190,190]
    elif rel_step < 2*n: #Move COM to leg 3 by making leg 3 smaller and adding pressure to chamber 2 of leg 2 and chamber 3 of leg 4
        d_pressures[l1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*180
        d_pressures[l2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*210
        d_pressures[l3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*180
        d_pressures[l4,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*210
        #[10,10,10,  400,400,400,    10,10,10,    400,400,400]
    elif rel_step < 3*n: # Lift leg 1 
        d_pressures[l1,p1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        d_pressures[l1,p3]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[185,10,360,  400,400,400,    10,10,10,    400,400,400]
    elif rel_step < 4*n: # Place leg 1 down
        d_pressures[l2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[l3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40        
        d_pressures[l4,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250   
        d_pressures[l1,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l1,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l1,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150     
        #[135,60,210,  150,150,150,    50,50,50,    150,150,150]
    elif rel_step < 5*n: # Place leg 1 down
        d_pressures[l3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100       
        d_pressures[l1,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l1,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*75
        d_pressures[l1,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        #[85,135,35,  150,150,150,    150,150,150,    150,150,150]
    elif rel_step < 5*n+1:
        d_pressures = np.zeros((4,3))
        #[85,135,35,  150,150,150,    150,150,150,    150,150,150]
    elif rel_step < 6*n: #Raise body to neutral position #210
        d_pressures[l1,p1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*125
        d_pressures[l1,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*75
        d_pressures[l1,p3]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        d_pressures[l2:,:] = (np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*60  
        #[210,210,210,  210,210,210,    210,210,210,    210,210,210]
    elif rel_step < 7*n: #Move COM to leg 1 by making leg 1 smaller and adding pressure to leg 2 and leg 4
        d_pressures[l1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200
        d_pressures[l2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*190
        d_pressures[l3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*200
        d_pressures[l4,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*190
        #[10,10,10,  400,400,400,    10,10,10,    400,400,400]
    elif rel_step < 8*n: # Lift leg 3
        d_pressures[l3,p1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        d_pressures[l3,p3]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[10,10,10,  400,400,400,    185,10,360,    400,400,400]
    elif rel_step < 9*n: # Place leg 3 down
        d_pressures[l2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[l1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40        
        d_pressures[l4,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250 
        d_pressures[l3,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l3,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l3,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150           
        #[50,50,50,  150,150,150,    135,60,210,    150,150,150]
    elif rel_step < 10*n: # Place leg 3 down
        d_pressures[l1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 
        d_pressures[l3,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l3,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*75
        d_pressures[l3,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        #[150,150,150,  150,150,150,    85,135,35,    150,150,150]
    elif rel_step < 10*n+1:
        d_pressures = np.zeros((4,3))
        #[150,150,150,  150,150,150,    85,135,35,    150,150,150]
    elif rel_step < 11*n: #Raise body to neutral position #230
        d_pressures[l3,p1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*145
        d_pressures[l3,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*95
        d_pressures[l3,p3]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*195
        d_pressures[:l3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*80  
        d_pressures[l4,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*80
        #[230,230,230,  230,230,230,    230,230,230,    230,230,230,]
    elif rel_step < 12*n: #Move COM to leg 4 by making leg 4 smaller and adding pressure to leg 1 and leg 3
        d_pressures[l1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*170
        d_pressures[l2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*220
        d_pressures[l3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*170
        d_pressures[l4,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*220
        #[400,400,400,  10,10,10,    400,400,400,    10,10,10,]
    elif rel_step < 13*n: # Lift leg 2
        d_pressures[l2,p1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        d_pressures[l2,p3]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[400,400,400,  185,10,360,    400,400,400,    10,10,10,]
    elif rel_step < 14*n: # Place leg 2 down
        d_pressures[l1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[l2,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l2,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l2,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[l3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250        
        d_pressures[l4,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40           
        #[150,150,150,  135,60,210,   150,150,150     50,50,50]
    elif rel_step < 15*n: # Place leg 2 down
        d_pressures[l2,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l2,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*75
        d_pressures[l2,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        d_pressures[l4,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100           
        #[150,150,150,   85,135,35,   150,150,150,    150,150,150]
    elif rel_step < 15*n+1:
        d_pressures = np.zeros((4,3))
        #[150,150,150,   85,135,35,   150,150,150,    150,150,150]
    elif rel_step < 16*n: #Raise body to neutral position #250
        d_pressures[l2,p1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*165
        d_pressures[l2,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*115
        d_pressures[l2,p3]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*215
        d_pressures[l1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100 
        d_pressures[l3:,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        #[250,250,250,  250,250,250,    250,250,250,    250,250,250]
    elif rel_step < 17*n: #Move COM to leg 2 by making leg 2 smaller and adding pressure to leg 1 and leg 3
        d_pressures[l1,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[l2,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*240
        d_pressures[l3,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150
        d_pressures[l4,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*240
        #[400,400,400,  10,10,10,    400,400,400,    10,10,10,]
    elif rel_step < 18*n: # Lift leg 4
        d_pressures[l4,p1]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        d_pressures[l4,p3]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*350
        #[400,400,400,  10,10,10,    400,400,400,    185,10,360,]
    elif rel_step < 19*n: # Place leg 4 down
        d_pressures[l1,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[l2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*40
        d_pressures[l3,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*250
        d_pressures[l4,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l4,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l4,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150        
        #[150,150,150,  50,50,50,   150,150,150     135,60,210]
    elif rel_step < 20*n: # Place leg 4 down
        d_pressures[l2,:]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*100
        d_pressures[l4,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*50
        d_pressures[l4,p2]=(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*75
        d_pressures[l4,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*175
        #[150,150,150,  150,150,150,    150,150,150,    85,135,35]
    elif rel_step < 20*n+1:
        d_pressures = np.zeros((4,3))
        #[150,150,150,  150,150,150,    150,150,150,    85,135,35]
    elif rel_step < 21*n: #Body to zero position #0
        d_pressures[:l4,:]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*150  
        d_pressures[l4,p1]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*85
        d_pressures[l4,p2]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*135
        d_pressures[l4,p3]=-(np.pi*dn*np.sin(np.pi*rel_rel_step*dn))*0.5*35
    return d_pressures

def demo_suction_rot(step, mode):
    gait_cycle = step // gait_length
    rel_step = step % gait_length
    suction_action = np.array([False,False,False,False])

    l1,l2,l3,l4 = 0,1,2,3
    
    if rel_step < 5*n:
        suction_action[l1] = False
        suction_action[l2] = True
        suction_action[l3] = True
        suction_action[l4] = True
    
    elif rel_step < 10*n:
        suction_action[l1] = True
        suction_action[l2] = True
        suction_action[l3] = False
        suction_action[l4] = True

    elif rel_step < 15*n:
        suction_action[l1] = True
        suction_action[l2] = False
        suction_action[l3] = True
        suction_action[l4] = True

    elif rel_step < 20*n:
        suction_action[l1] = True
        suction_action[l2] = True
        suction_action[l3] = True
        suction_action[l4] = False
    
    else:
        suction_action[l1] = False
        suction_action[l2] = True
        suction_action[l3] = True
        suction_action[l4] = True
    
    return suction_action
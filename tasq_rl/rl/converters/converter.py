import numpy as np
deg2rad = np.pi/180 #conversion factor from degrees to rad

theta = np.pi/6 # positioning angle for hall sensors in rad
hs_radius = 0.0065 # positioning radius for hall sensors in m

def pneu2posandgains(pressures, no_links):
    """
    k1 = 0.0003 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
    k2 = 0.0000005 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
    kp__th_1 = 0.08# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
    kd__th_1 = 0.1# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
    kp__dz_1 = 0.15# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
    kd__dz_1 = 0.1#velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)
    c1 = 2 # constant to map torque to change in position gain for revolute joints
    c2 = 1 # constant to map torque to change in velocity gain for revolute joints
    c3 = 2 # constant to map force to change in position gain for prismatic joints
    c4 = 1 # constant to map force to change in velocity gain for prismatic joints
    """
    k1 = 0.000026 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
    k2 = 0.00000022 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
    kp__th_1 = 0.08# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
    kd__th_1 = 0.2# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
    kp__dz_1 = 0.15# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
    kd__dz_1 = 0.2# velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)
    c1 = 2 # constant to map torque to change in position gain for revolute joints
    c2 = 1 # constant to map torque to change in velocity gain for revolute joints
    c3 = 2 # constant to map force to change in position gain for prismatic joints
    c4 = 1 # constant to map force to change in velocity gain for prismatic joints

    if no_links == 3:
        k1 = k1/3 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
        k2 = k2/3 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
        kp__th_1 = kp__th_1/3# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
        kd__th_1 = kd__th_1/3# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
        kp__dz_1 = kp__dz_1/3# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
        kd__dz_1 = kd__dz_1/3# velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)

    if no_links == 4:
        k1 = k1/2.5 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
        k2 = k2/2.5 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
        kp__th_1 = kp__th_1/2.5# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
        kd__th_1 = kd__th_1/2.5# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
        kp__dz_1 = kp__dz_1/2.5# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
        kd__dz_1 = kd__dz_1/2.5# velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)

    if no_links == 5:
        k1 = k1/2 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
        k2 = k2/2 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
        kp__th_1 = kp__th_1/2# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
        kd__th_1 = kd__th_1/2# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
        kp__dz_1 = kp__dz_1/2# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
        kd__dz_1 = kd__dz_1/2# velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)

    if no_links == 6:
        k1 = k1/1.5 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
        k2 = k2/1.5 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
        kp__th_1 = kp__th_1/1.5# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
        kd__th_1 = kd__th_1/1.5# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
        kp__dz_1 = kp__dz_1/1.5# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
        kd__dz_1 = kd__dz_1/1.5# velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)
    """
    if no_links == 7:
        k1 = k1 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
        k2 = k2 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
        kp__th_1 = kp__th_1# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
        kd__th_1 = kd__th_1# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
        kp__dz_1 = kp__dz_1# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
        kd__dz_1 = kd__dz_1# velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)
    """

    pressure1, pressure2, pressure3 = pressures
    # Base vectors
    vx = np.array([1, 0, 0])
    vy = np.array([0, 1, 0])
    vz = np.array([0, 0, 1])
    # Pressure vectors
    p1 = pressure1*vz
    p2 = pressure2*vz
    p3 = pressure3*vz
    # Pressure acting point vector
    v1 = hs_radius*vx
    v2 = hs_radius*(-np.sin(theta) * vx - np.cos(theta) * vy)
    v3 = hs_radius*(-np.sin(theta) * vx + np.cos(theta) * vy)
    # Resulting torque and force
    T = k1*(np.cross(p1,v1) + np.cross(p2,v2) + np.cross(p3,v3)) *no_links
    F = k2*(p1 + p2 + p3) *no_links
    # Feedback gains 
    #random_noise_gains = np.random.normal(1, 0.01, size=(4,))
    kp_thx_n = (kp__th_1 + np.abs(T[0]) * c1) * no_links
    kd_thx_n = (kd__th_1 + np.abs(T[0]) * c2) * no_links 
    kp_thy_n = (kp__th_1 + np.abs(T[1]) * c1) * no_links
    kd_thy_n = (kd__th_1 + np.abs(T[1]) * c2) * no_links
    kp_dz_n = (kp__dz_1 + F[2] * c3) * no_links 
    kd_dz_n = (kd__dz_1 + F[2] * c4) * no_links 
    # Target angles and position
    #random_noise_pos = np.random.normal(1, 0.01, size=(3,))
    thx = T[0]/(kp_thx_n * deg2rad)
    thy = T[1]/(kp_thy_n * deg2rad)
    dz = -F[2]/kp_dz_n
    
    thx = 0
    thy = 0
    dz = 0
    
    kp_thx_n = 0.01
    kd_thx_n = 2
    kp_thy_n = 0.01
    kd_thy_n = 2
    kp_dz_n = 0.1
    kd_dz_n = 1.8
    

    
    """
    print("\n=== Computed Target Joint Positions ===")
    print(f"{'Axis':<5} | {'Target Pos':>12}")
    print("-" * 22)
    print(f"{'X':<5} | {thx:12.4f}")
    print(f"{'Y':<5} | {thy:12.4f}")
    print(f"{'Z':<5} | {dz:12.8f}")
    print()
    """

    return (
        thx, thy, dz,
        kp_thx_n, kd_thx_n,
        kp_thy_n, kd_thy_n,
        kp_dz_n, kd_dz_n
    )
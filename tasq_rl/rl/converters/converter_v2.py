import numpy as np
deg2rad = np.pi/180 #conversion factor from degrees to rad

theta = np.pi/6 # positioning angle for hall sensors in rad
hs_radius = 0.0065 # positioning radius for hall sensors in m

k1 = 0.0002 # constant to map pressure to torque (physical interpretation: area to which pressure is applied)
k2 = 0.0000005 # constant to map pressure to force (physical interpretation: area to which pressure is applied)
kp__th_1 = 0.008# position gain for revolute joints when no pressure is applied (physical interpretation: bending stiffness)
kd__th_1 = 0.1# velocity gain for revolute joints when no pressure is applied (physical interpretation: bending damping)
kp__dz_1 = 0.05# position gain for prismatic z joints when no pressure is applied (physical interpretation: axial stiffness)
kd__dz_1 = 0.2# velocity gain for prismatic z joints when no pressure is applied (physical interpretation: axial damping)
c1 = 2 # constant to map torque to change in position gain for revolute joints
c2 = 1 # constant to map torque to change in velocity gain for revolute joints
c3 = 20 # constant to map force to change in position gain for prismatic joints
c4 = 10 # constant to map force to change in velocity gain for prismatic joints

def pneu2posandgains(pressures, no_links):
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
    T = k1*(np.cross(p1,v1) + np.cross(p2,v2) + np.cross(p3,v3)) 
    F = k2*(p1 + p2 + p3)
    # Feedback gains 
    kp_thx_n = (kp__th_1 + np.abs(T[0]) * c1) * no_links 
    kd_thx_n = (kd__th_1 + np.abs(T[0]) * c2) * no_links
    kp_thy_n = (kp__th_1 + np.abs(T[1]) * c1) * no_links
    kd_thy_n = (kd__th_1 + np.abs(T[1]) * c2) * no_links
    kp_dz_n = (kp__dz_1 + F[2] * c3) * no_links
    kd_dz_n = (kd__dz_1 + F[2] * c4) * no_links
    # Target angles and position
    thx = T[0]/(kp_thx_n * deg2rad)
    thy = T[1]/(kp_thy_n * deg2rad)
    dz = -F[2]/kp_dz_n

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
import numpy as np
#from ... import #models for converting pressure to angles, elongation and stiffness and damping constants and vice versa

deg2rad = np.pi/180 #conversion factor from degrees to rad
c = 0.2 # constant to map pressures to angle
d = 0.00005 # constant to map pressures to distance

#n=1
e = 0.00000001 # constant to map pressures to gains
f = 0.001 # constant to map pressures to gains
g = 0.00005 # constant to map pressures to gains
h = 0.008 # constant to map pressures to gains
k = 0.0000000005 # constant to map pressures to gains
l = 0.000075 # constant to map pressures to gains
m = 0.000000005 # constant to map pressures to gains
n = 0.000075 # constant to map pressures to gains
#n=7 (no multiplication with no_links in gain calculation)
"""
f = 0.01 # constant to map pressures to gains
h = 0.025 # constant to map pressures to gains
l = 0.5 # constant to map pressures to gains
n = 0.5 # constant to map pressures to gains
"""
#n=3 (no multiplication with no_links in gain calculation)
"""
f = 0.0001 # constant to map pressures to gains
h = 0.0005 # constant to map pressures to gains
l = 0.000001 # constant to map pressures to gains
n = 0.000001 # constant to map pressures to gains
"""

theta = np.pi/6 # positioning angle for hall sensors in rad
hs_radius = 0.0065 # positioning radius for hall sensors in m

## Convert actuator pressures to target positions and gains for simulation
def pneu2posandgains(pressures, no_links):
    """
    Converts actuator pressures to target joint positions and control gains.

    Args:
        pressures (list or array): Pressures from the 3 actuators [mBar].
        no_links (int): Number of robot links.

    Returns:
        tuple: Target positions and gains in the following order:
               (target_pos_x, target_pos_y, target_pos_z,
                pos_gain_x, vel_gain_x,
                pos_gain_y, vel_gain_y,
                pos_gain_z, vel_gain_z)
    """
    pressure1, pressure2, pressure3 = pressures
    # Base vectors
    vx = np.array([1, 0])
    vy = np.array([0, 1])
    v1 = vx
    v2 = -np.sin(theta) * vx - np.cos(theta) * vy
    v3 = -np.sin(theta) * vx + np.cos(theta) * vy

    # Resulting vector and mean pressure
    v_res = pressure1 * v1 + pressure2 * v2 + pressure3 * v3
    #print(v_res)
    mean_pressure = np.mean(pressures)

    # Angles (in radians) and elongation (in meters)
    angle_x = -v_res[1] * c * deg2rad
    angle_y = v_res[0] * c * deg2rad
    #print(angle_x, angle_y)
    d_z = mean_pressure * -d

    # Target joint positions
    target_pos_x = angle_x / no_links
    target_pos_y = angle_y / no_links
    target_pos_z = d_z / (2 * no_links)
    
    """
    print("\n=== Computed Target Joint Positions ===")
    print(f"{'Axis':<5} | {'Target Pos':>12}")
    print("-" * 22)
    print(f"{'X':<5} | {target_pos_x:12.4f}")
    print(f"{'Y':<5} | {target_pos_y:12.4f}")
    print(f"{'Z':<5} | {target_pos_z:12.8f}")
    print()
    """
    
    # Control gains
    pos_gain_x = (e * np.abs(v_res[1]) * mean_pressure +f) * no_links**2
    pos_gain_y = (e * np.abs(v_res[0]) * mean_pressure +f) * no_links**2
    pos_gain_z = (g * mean_pressure + h) * no_links**2

    vel_gain_x = (k * np.abs(v_res[1]) * mean_pressure + l) * no_links**4.5
    vel_gain_y = (k * np.abs(v_res[0]) * mean_pressure + l) * no_links**4.5
    vel_gain_z = (m * mean_pressure +n) * no_links**4.5
    
    """
    print("\n=== Computed Joint Gains ===")
    print(f"{'Axis':<5} | {'Pos Gain':>10} | {'Vel Gain':>10}")
    print("-" * 33)
    print(f"{'X':<5} | {pos_gain_x:10.6f} | {vel_gain_x:10.6f}")
    print(f"{'Y':<5} | {pos_gain_y:10.6f} | {vel_gain_y:10.6f}")
    print(f"{'Z':<5} | {pos_gain_z:10.6f} | {vel_gain_z:10.6f}")
    print()
    """

    return (
        target_pos_x, target_pos_y, target_pos_z,
        pos_gain_x, vel_gain_x,
        pos_gain_y, vel_gain_y,
        pos_gain_z, vel_gain_z
    )

# Convert simulation target positions back to actuator pressures
def posandgains2pneu(target_pos_x, target_pos_y, target_pos_z, no_links):
    """
    Converts simulation target joint positions to actuator pressures.

    Args:
        target_pos_x (float): Target angle X [rad/link].
        target_pos_y (float): Target angle Y [rad/link].
        target_pos_z (float): Target elongation Z [m/link].
        no_links (int): Number of robot links.

    Returns:
        tuple: Actuator pressures (pressure1, pressure2, pressure3) [kPa].
    """
    # Reconstruct total angles and pressure
    angle_x = target_pos_x * no_links
    angle_y = target_pos_y * no_links
    mp = target_pos_z * (2 * no_links) / d

    # Reconstruct the resulting vector
    v_res = np.array([
        angle_y / (c * deg2rad),
        angle_x / (c * deg2rad)
    ])

    v1 = v_res[1]
    v2 = v_res[0]

    # Calculate actuator pressures
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    pressure1 = (3 * mp * sin_theta + v1) / (sin_theta + 1)
    pressure2 = (3 * mp - v1 - v2 * tan_theta - v2 / cos_theta) / (2 * (sin_theta + 1))
    pressure3 = (v2 + (3 * mp - v1 + v2 * tan_theta) * cos_theta) / (2 * (sin_theta + 1) * cos_theta)

    return pressure1, pressure2, pressure3

# Convert torques to forces sensed at the suction cup
def torque2force(tx, ty, fp):
    """
    Converts torque values to the corresponding forces at suction cup.

    Args:
        tx (float): Torque around X-axis [Nm].
        ty (float): Torque around Y-axis [Nm].
        fp (float): Force along Z-axis [N].

    Returns:
        tuple: Forces from actuators (f1, f2, f3) [N].
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    f1 = (fp * hs_radius * sin_theta - ty) / (hs_radius * (sin_theta + 1))
    f2 = (fp * hs_radius - tx * tan_theta - tx / cos_theta + ty) / (2 * hs_radius * (sin_theta + 1))
    f3 = (tx + (fp * hs_radius + tx * tan_theta + ty) * cos_theta) / (2 * hs_radius * (sin_theta + 1) * cos_theta)

    return f1, f2, f3
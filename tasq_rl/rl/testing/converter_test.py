import numpy as np

from rl.converters.converter import pneu2posandgains 

target_pos_x, target_pos_y, target_pos_z, pos_gain_x, vel_gain_x, pos_gain_y, vel_gain_y, pos_gain_z, vel_gain_z = pneu2posandgains(np.array([400,0,0]),7)
#print(target_pos_x, target_pos_y, target_pos_z, pos_gain_x, vel_gain_x, pos_gain_y, vel_gain_y, pos_gain_z, vel_gain_z)

print("\n=== Computed Target Joint Positions ===")
print(f"{'Axis':<5} | {'Target Pos':>12}")
print("-" * 22)
print(f"{'X':<5} | {target_pos_x:12.4f}")
print(f"{'Y':<5} | {target_pos_y:12.4f}")
print(f"{'Z':<5} | {target_pos_z:12.8f}")
print()

print("\n=== Computed gains ===")
print(f"{'Axis':<5} | {'Gains':>12}")
print("-" * 22)
print(f"{'X':<5} | {pos_gain_x:12.8f} | {vel_gain_x:12.8f}")
print(f"{'Y':<5} | {pos_gain_y:12.8f} | {vel_gain_y:12.8f}")
print(f"{'Z':<5} | {pos_gain_z:12.8f} | {vel_gain_z:12.8f}")
print()
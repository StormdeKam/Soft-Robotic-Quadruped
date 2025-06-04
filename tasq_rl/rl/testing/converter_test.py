import numpy as np

from converter.converter_fn import pneu2posandgains 

target_pos_x, target_pos_y, target_pos_z, pos_gain_x, vel_gain_x, pos_gain_y, vel_gain_y, pos_gain_z, vel_gain_z = pneu2posandgains(np.array([10,100,100]),11)
print(target_pos_x, target_pos_y, target_pos_z, pos_gain_x, vel_gain_x, pos_gain_y, vel_gain_y, pos_gain_z, vel_gain_z)


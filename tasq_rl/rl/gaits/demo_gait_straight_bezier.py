from rl.utils.bezier_gait_planner import BezierGaitPlanner
import numpy as np

planner = BezierGaitPlanner(num_channels=12)

no_channels = 12
# Define gait curve for channel 0
def sine(channels,start_amplitude, amplitude, start_time, length, period_start, period, num_points):
    for channel in channels:
        for i in range(num_points):
            planner.add_point(channel, start_time + (i+1)*length/num_points, start_amplitude + amplitude*np.sin(period_start+(i+1)*period/num_points))

def single_point(channels, time, pressure):
    for channel in channels:
        planner.add_point(channel, time, pressure)

sine([0,1,2,3,4,5,6,7,8,9,10,11], 0, 200, 0, 1, 0, np.pi/2,50)
sine([0,1,2,6,7,8], 200, -190, 1, 1, 0, np.pi/2,50)
sine([3,4,5,9,10,11], 200, 190, 1, 1, 0, np.pi/2,50)
single_point([3,4,5,9,10,11], 3, 390)
single_point([1,2], 2+(1/6), 10)
single_point([2], 2+(1/3), 10)
sine([0], 10, 380, 2, 1, 0, np.pi,8)
sine([1], 10, 380, 2+(1/3), 1, 0, np.pi,8)
sine([2], 10, 380, 2+(2/3), 1, 0, np.pi,8)
"""
for channel in range(no_channels):
    planner.add_point(channel, 0, 0)
    planner.add_point(channel, 0.25, 200*np.sin(np.pi/8))
    planner.add_point(channel, 0.5, 200*np.sin(np.pi/4))
    planner.add_point(channel, 0.75, 200*np.sin(3*np.pi/8))
    planner.add_point(channel, 1, 200*np.sin(np.pi/2))
for channel in [0,1,2,6,7,8]:
    planner.add_point(channel, 1.25, 200-190*np.sin(np.pi/8))
    planner.add_point(channel, 1.5, 200-190*np.sin(np.pi/4))
    planner.add_point(channel, 1.75, 200-190*np.sin(3*np.pi/8))
    planner.add_point(channel, 2, 200-190*np.sin(np.pi/2))
for channel in [3,4,5,9,10,11]:
    planner.add_point(channel, 1.25, 200+190*np.sin(np.pi/8))
    planner.add_point(channel, 1.5, 200+190*np.sin(np.pi/4))
    planner.add_point(channel, 1.75, 200+190*np.sin(3*np.pi/8))
    planner.add_point(channel, 2, 200+190*np.sin(np.pi/2))
for channel in [3,4,5,9,10,11]:
    planner.add_point(channel, 3, 390)
for channel in [0,1,2]:
    planner.add_point(channel, 1.25, 200-190*np.sin(np.pi/8))
    planner.add_point(channel, 1.5, 200-190*np.sin(np.pi/4))
    planner.add_point(channel, 1.75, 200-190*np.sin(3*np.pi/8))
    planner.add_point(channel, 2, 200-190*np.sin(np.pi/2))
"""





# Plot the full curves
planner.plot(duration=10.0, dt=0.005)

from rl.scripts.tasq_simulator import TASQSimulator
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Run TASQ with Behavior Cloning or SAC")
parser.add_argument("--gait", type=str, choices=["straight_bezier", "straight", "straightv3", "rotation"], required=True, help="Run straight_bezier, straight or rotation gait")
parser.add_argument("--no_links", type=str, choices=["5","6","7"], required=True, help="amount of links in the robot legs")
args = parser.parse_args()

no_links = int(args.no_links)
gait_length = 1400*2

urdf_dict = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}

if args.gait == "straight_bezier":
    from rl.gaits.demo_gait_straight_bezier import DemoGaitStraight
    demogait = DemoGaitStraight()
    demogait.gait()
    demo_pressures = demogait.demo_pressures
    demo_suction = demogait.demo_suction
    #demogait.plot()
    sim = TASQSimulator(urdf_dict[no_links], gui=True, no_links = no_links)
    sim.reset_simulation()
    sim.run(steps=10*gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction, mode=0)

elif args.gait == "straightv3":
    from rl.gaits.old.demo_gait_straight_v3 import gait_pressures, gait_suction
    sim = TASQSimulator("rl/urdfs/TASQ_n5.urdf", gui=True, no_links=5)
    sim.reset_simulation()
    sim.run(steps=10*gait_length, pressure_callback=gait_pressures, suction_callback=gait_suction, mode=0)
    
elif args.gait == "straight":
    from rl.gaits.demo_gait_straight_low_freq import demo_pressures, demo_suction
    demo_pressures_scaled = lambda step, mode: demo_pressures(step, mode) / 100
    demo_suction_scaled = lambda step, mode: demo_suction(step, mode) / 100
    sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
    sim.reset_simulation()
    sim.run(steps=1400,pressure_callback=demo_pressures_scaled,suction_callback=demo_suction_scaled, mode=0)

elif args.gait == "rotation":
    from rl.gaits.old.demo_gait_rotation import demo_pressures_rot as demo_pressures
    from rl.gaits.old.demo_gait_rotation import demo_suction_rot as demo_suction
    sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
    sim.reset_simulation()
    sim.run(steps=4*gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction, mode=0)

sim.disconnect()







from rl.scripts.tasq_simulator import TASQSimulator
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Run TASQ with Behavior Cloning or SAC")
parser.add_argument("--gait", type=str, choices=["straight_bezier", "straight", "straightv3", "rotation"], required=True, help="Run straight_bezier, straight or rotation gait")
args = parser.parse_args()


gait_length = 10*23



if args.gait == "straight_bezier":
    from rl.gaits.demo_gait_straight_bezier import DemoGaitStraight
    demogait = DemoGaitStraight()
    demogait.gait()
    demo_pressures = demogait.demo_pressures
    demo_suction = demogait.demo_suction
    demogait.plot()
    sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
    sim.reset_simulation()
    sim.run(steps=4*gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction, mode=0)

elif args.gait == "straightv3":
    from rl.gaits.demo_gait_straight_v3 import gait_pressures, gait_suction
    sim = TASQSimulator("rl/urdfs/TASQ_n5.urdf", gui=True, no_links=5)
    sim.reset_simulation()
    sim.run(steps=10*gait_length, pressure_callback=gait_pressures, suction_callback=gait_suction, mode=0)
    
elif args.gait == "straight":
    from rl.gaits.demo_gait_straight import demo_pressures, demo_suction
    sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
    sim.reset_simulation()
    sim.run(steps=4*gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction, mode=0)

elif args.gait == "rotation":
    from rl.gaits.demo_gait_rotation import demo_pressures_rot as demo_pressures
    from rl.gaits.demo_gait_rotation import demo_suction_rot as demo_suction
    sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
    sim.reset_simulation()
    sim.run(steps=4*gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction, mode=0)

sim.disconnect()







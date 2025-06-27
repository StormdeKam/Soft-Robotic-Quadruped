from rl.scripts.tasq_simulator import TASQSimulator
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Run TASQ with Behavior Cloning or SAC")
parser.add_argument("--gait", type=str, choices=["straight", "rotation"], required=True, help="Run straight or rotation gait")
args = parser.parse_args()


gait_length = 10*23


if args.gait == "straight":
    from rl.gaits.demo_gait_straight import demo_pressures, demo_suction
elif args.gait == "rotation":
    from rl.gaits.demo_gait_rotation import demo_pressures_rot as demo_pressures
    from rl.gaits.demo_gait_rotation import demo_suction_rot as demo_suction

sim = TASQSimulator("rl/urdfs/TASQ_n7.urdf", gui=True, no_links=7)
sim.reset_simulation()
sim.run(steps=gait_length, pressure_callback=demo_pressures, suction_callback=demo_suction, mode=1)
sim.disconnect()







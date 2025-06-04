import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot_id = p.loadURDF("tasq_rl/urdfs/TASQ_n3.urdf",[0,0,2])

print("Robot ID:", robot_id)
print("Num bodies:", p.getNumBodies())

for i in range(10):
    p.stepSimulation()
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    print("Base pos:", pos, "orn:", orn)
    time.sleep(0.1)

p.disconnect()

import pybullet as p
import pybullet_data
import time

# Start PyBullet with GUI (you can also use DIRECT for headless)
physicsClient = p.connect(p.GUI)

# Set the path to PyBullet's example data
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load plane and robot
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("r2d2.urdf", [0, 0, 1])

# Set gravity
p.setGravity(0, 0, -9.8)

# Run the simulation for a few seconds
for _ in range(240):
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect from the simulation
p.disconnect()

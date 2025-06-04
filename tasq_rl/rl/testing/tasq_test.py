import pybullet as p
import time
import pybullet_data
import numpy as np

path = r"/home/storm-de-kam/Documents/tasq_rl/urdfs/urdfv3/urdf/TASQ_n11.urdf"

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0.2]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF(path,startPos, startOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

revolute_joints = []
prismatic_joints = []
# Get the number of joints in the model
num_joints = p.getNumJoints(boxId)

# Control gains
k_p = 0.4  # Stiffness
k_d = 0.00002  # Damping
max_torque = 0.01
position_gain = 0.5  # 0.0–1.0
velocity_gain = 1.0  # typically 1.0

for joint_index in range(num_joints):
    joint_info = p.getJointInfo(boxId, joint_index)
    joint_type = joint_info[2]
    if joint_type in [p.JOINT_REVOLUTE]:
        revolute_joints.append(joint_index)
        p.setJointMotorControl2(
            bodyIndex=boxId,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            positionGain=position_gain,
            velocityGain=velocity_gain,
            force=500  # Max motor force (acts like torque limit)
        )
    if joint_type in [p.JOINT_PRISMATIC]:
    	prismatic_joints.append(joint_index)
        p.setJointMotorControl2(
            bodyIndex=boxId,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            positionGain=position_gain,
            velocityGain=velocity_gain,
            force=500  # Max motor force (acts like torque limit)
    

# Define trajectory parameters
total_time = 5.0  # seconds
time_step = 1.0 / 240.0
steps = int(total_time / time_step)
target_angle_rad = np.deg2rad(-5)  # convert degrees to radians

# Precompute trajectory: linear interpolation from 0 to 2 degrees
trajectory = np.linspace(0, target_angle_rad, steps)


p.changeDynamics(
    bodyUniqueId=boxId,
    linkIndex=24,             # Use -1 for base link
    lateralFriction=0.99       # New friction value
)

def apply_attraction_force(bodyA, linkA, bodyB, linkB, force_magnitude=5.0):
    #"""Apply an attracting force from bodyB:linkB to bodyA:linkA."""
    # Get positions
    if linkA == -1:
        posA, _ = p.getBasePositionAndOrientation(bodyA)
    else:
        posA = p.getLinkState(bodyA, linkA)[0]

    if linkB == -1:
        posB, _ = p.getBasePositionAndOrientation(bodyB)
    else:
        posB = p.getLinkState(bodyB, linkB)[0]

    # Compute direction and apply force
    direction = [0,0,posB[2] - posA[2]]
    distance = sum(d ** 2 for d in direction) ** 0.5
    if distance > 0 and distance < 0.01:
        norm_direction = [d / distance for d in direction]
        force = [force_magnitude * d for d in norm_direction]
        p.applyExternalForce(
            objectUniqueId=bodyA,
            linkIndex=linkA,
            forceObj=force,
            posObj=posA,
            flags=p.WORLD_FRAME
        )

# Run the simulation
for _ in range(10000):
    # Apply attraction from plane (base link -1) to robot link 24
    #apply_attraction_force(boxId, 23, planeId, -1, force_magnitude=20.0)

    for i, j in enumerate(controlled_joints):
            p.setJointMotorControl2(
                bodyIndex=boxId,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                positionGain=position_gain,
                velocityGain=velocity_gain,
                force=500  # Max motor force (acts like torque limit)
            )

    """
    if _ < 1200:
        for i, j in enumerate(activated_joints):
            p.setJointMotorControl2(
                bodyIndex=boxId,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=trajectory[_],
                positionGain=position_gain,
                velocityGain=velocity_gain,
                force=500  # Max motor force (acts like torque limit)
            )
    """
    

    p.stepSimulation()
    time.sleep(1./240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()

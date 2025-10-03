import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import logging
from rl.converters.converter import pneu2posandgains

class TASQSimulator:
    def __init__(self, urdf_path, gui=False, no_links=0, render_mode=None):
        #self.start_yaw = np.random.uniform(-np.pi, np.pi)
        self.start_yaw = np.pi/4
        self.no_links = no_links
        self.urdf_path = urdf_path
        self.gui = gui
        self.render_mode = render_mode  # Not used, but keeps Gymnasium happy
        # Ensure no existing connection
        try:
            p.disconnect()
        except Exception:
            pass
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())


        #change background color to black
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.data_logging = False
        self.log_data = []  # Buffer for logging pressure + suction
        self._current_pressures = None  # Temporarily holds last pressures
        self.randomization = True

    def reset_simulation(self):
        # Reset PyBullet simulation (clears all bodies, constraints, etc.)
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        
        # Reset camera as in constructor
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw= 180,
                cameraPitch=-1,
                cameraTargetPosition=[0, 0, 0.1],
                physicsClientId=self.physics_client
            )
        
        # Reload plane and robot
        self._load_environment()
        
        # Reset robot internals, joints, gains, etc.
        self._initialize_robot()

        # Reset internal state variables
        #self.suction_states = np.array([False, True, True, True])
        #self.pressure_state = np.zeros((4, 3))
        self.log_data.clear()
        self._current_pressures = None
        
        # Reset other counters or flags if you have any
        # e.g., self.step_count = 0

    def _load_environment(self):
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])  # grey
        self.start_x = np.random.uniform(0.1, 0.1)
        self.start_y = np.random.uniform(0.1, 0.1)
        self.start_x, self.start_y = 0.0, 0.0
        self.start_pos = [self.start_x, self.start_y, 0.16]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, self.start_yaw])
        self.robot_id = p.loadURDF(self.urdf_path, self.start_pos, self.start_orientation)
            
    def _initialize_robot(self):
        self.revolute_joints = []
        self.prismatic_joints = []
        self.suction_states = np.array([0,100,100,100],dtype=np.float64)  # Initialize suction states
        self.pressure_state = np.ones((4,3)) *10 
        self.last_targets = np.zeros((4,3)) *10 
        self.control_freq = 24 #Hz

        self.num_joints = p.getNumJoints(self.robot_id)
        # Domain randomization:
        if self.randomization:
            self._randomize_link_masses(mass_range=(0.8, 1.2))  # ±0% mass
            self.joint_randomizer_values = np.random.normal(1, 0.001, size=(6,))
            self.suction_randomization = np.random.normal(1, 0.2, size=(1,))
            self.hysteresis_randomization = np.random.normal(1, 0.1, size=(4,))
        else:
            self.joint_randomizer_values = np.ones((6,))
            self.suction_randomization = 1.0
            self.hysteresis_randomization = np.ones(4)

        self.position_gain = 1
        self.velocity_gain = 1.2
        self.max_force = 500
        self.max_lin_vel=1000
        self.max_ang_vel=1000
        self.max_vel = 1000

        excluded_names = {"j1_ps", "j2_ps", "j3_ps", "j4_ps"}
        
        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode()
            joint_type = joint_info[2]

            # set leg to body spherical joints to very stiff position control
            if joint_type == p.JOINT_SPHERICAL:
                p.setJointMotorControlMultiDof(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],   # quaternion (w, x, y, z)
                positionGain=1,                  # proportional gain (same on all axes)
                velocityGain=1.4,                   # damping gain (same on all axes)
                force=[self.max_force,self.max_force,self.max_force],             # torque limits per axis
                #maxVelocity=self.max_lin_vel
                )
                

            if joint_type == p.JOINT_REVOLUTE:
                self.revolute_joints.append(joint_index)
                #p.resetJointState(self.robot_id, joint_index, targetValue)
                #print(f"{joint_type} joint {joint_index}: {joint_name}, active")
            elif joint_type == p.JOINT_PRISMATIC and joint_name not in excluded_names:
                self.prismatic_joints.append(joint_index)
                #print(f"{joint_type} joint {joint_index}: {joint_name}, active")
            
        self.controlled_joints = self.revolute_joints + self.prismatic_joints
        
        for joint_index in self.controlled_joints:
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                positionGain=self.position_gain,
                velocityGain=self.velocity_gain,
                force=self.max_force,
                #maxVelocity=self.max_lin_vel
            )
        
        # Foot link indices
        self.foot_link_indices = np.zeros(4)
        for leg_idx in range(4):
            cup_link_idx = leg_idx*(self.no_links*4+2) + self.no_links*4 +1
            self.foot_link_indices[leg_idx] = cup_link_idx  # list of 4 ints

    def _randomize_link_masses(self, mass_range=(0.8, 1.2)):
        """
        Randomly scales the mass of each link for domain randomization.
        mass_range: tuple (min_scale, max_scale) to multiply the original link mass.
        """
        # Get the base mass first
        info = p.getDynamicsInfo(self.robot_id, -1)
        base_mass = info[0]  # mass is always the first element

        base_scale = np.random.normal(*mass_range)
        p.changeDynamics(self.robot_id, -1, mass=base_mass * base_scale)

        for joint_index in range(self.num_joints):
            # Each link is associated with joint_index
            info = p.getDynamicsInfo(self.robot_id, joint_index)
            mass = info[0]  # mass is always the first element

            scale = np.random.uniform(*mass_range)
            p.changeDynamics(self.robot_id, joint_index, mass=2* mass * scale)
    
    """
    def apply_pressures(self, pressures, pressure_gradients=np.ones((4,3))):
        self.pressure_state = pressures

        kp_th =0.1
        kd_th = 0.003
        kp_z = 10000
        kd_z = 0.1

        max_pressure = 400  # mBar

        if len(pressures) != 4 or any(len(leg) != 3 for leg in pressures):
            raise ValueError("Pressures must be a list of 4 arrays, each with 3 pressure values.")

        # Assume revolute and prismatic joints are grouped by leg
        revolutes_per_leg = len(self.revolute_joints) // 4
        prismatics_per_leg = len(self.prismatic_joints) // 4

        for leg_index, leg_pressures in enumerate(pressures):

            if leg_pressures.all() <= 400:
                Tx, Ty, F = pneu2posandgains(leg_pressures, self.no_links)

            else:
                self.logger.warning(f"Pressure in leg {leg_index+1} exceeding max_pressure of {max_pressure} mBar")

            # -------------------------
            # Apply to joints
            # -------------------------
            for j in range(revolutes_per_leg-2):
                joint_index = self.revolute_joints[leg_index * revolutes_per_leg + j]
                # PD control
                joint_state = p.getJointState(self.robot_id, joint_index)
                pos = joint_state[0]
                vel = joint_state[1]
                
                if j % 2 == 0:
                    torque = kp_th * (0 - pos) - kd_th* vel + Tx*500
                else:
                    torque = kp_th * (0 - pos) - kd_th* vel + Ty*500

                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.TORQUE_CONTROL,
                    force=torque
                )

            cup_link_idx = [leg_index*(self.no_links*4+2) + self.no_links*4 + 1]
            for j in range(prismatics_per_leg):
                joint_index = self.prismatic_joints[leg_index * prismatics_per_leg + j]
                if joint_index not in cup_link_idx:
                    joint_state = p.getJointState(self.robot_id, joint_index)
                    pos = joint_state[0]
                    vel = joint_state[1]
                    force = kp_z * (0 - pos) - kd_z * vel + F*10000
                    #print(pos)
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_index,
                        controlMode=p.TORQUE_CONTROL,
                        force=force
                    )

        self._current_pressures = np.array(pressures).flatten().tolist()  # Save for logging
    """            
    def apply_pressures(self, pressures, pressure_gradients = np.ones((4,3))):
        self.pressure_state = pressures
        #print(pressures)
        alpha = 0
        alpha_up = 0.2
        alpha_down = 0.02
        """
        Apply actuator pressures to the robot. Converts them to joint target positions/gains.

        :param pressures: list of 4 arrays, each with 3 pressure values (kPa),
                        e.g., [[x1, y1, z1], ..., [x4, y4, z4]]
        """

        max_pressure = 400 #mBar

        if len(pressures) != 4 or any(len(leg) != 3 for leg in pressures):
            raise ValueError("Pressures must be a list of 4 arrays, each with 3 pressure values.")

        # Assume revolute and prismatic joints are grouped by leg
        revolutes_per_leg = len(self.revolute_joints) // 4
        prismatics_per_leg = len(self.prismatic_joints) // 4
        
        for leg_index, leg_pressures in enumerate(pressures):

            if leg_pressures.all() <= 400:                
                target_pos_x, target_pos_y, target_pos_z, \
                pos_gain_x, vel_gain_x, \
                pos_gain_y, vel_gain_y, \
                pos_gain_z, vel_gain_z = pneu2posandgains(leg_pressures, self.no_links)
                
                target_pos_x *= self.joint_randomizer_values[0]
                target_pos_y *= self.joint_randomizer_values[0]
                target_pos_z *= self.joint_randomizer_values[1]
                pos_gain_x *= self.joint_randomizer_values[2]
                vel_gain_x *= self.joint_randomizer_values[3]
                pos_gain_y *= self.joint_randomizer_values[2]
                vel_gain_y *= self.joint_randomizer_values[3]
                pos_gain_z *= self.joint_randomizer_values[4]
                vel_gain_z *= self.joint_randomizer_values[5]          
            else:
                self.logger.warning(f"Pressure in leg  {leg_index +1} exceeding max_pressure of {max_pressure} mBar")

            if target_pos_x > self.last_targets[leg_index,0]:
                alpha = alpha_up*self.hysteresis_randomization[0]
            else:
                alpha = alpha_down*self.hysteresis_randomization[1]
            thx = self.last_targets[leg_index,0] + alpha * (target_pos_x - self.last_targets[leg_index,0])

            if target_pos_y > self.last_targets[leg_index,1]:
                alpha = alpha_up*self.hysteresis_randomization[0]
            else:
                alpha = alpha_down*self.hysteresis_randomization[1]
            thy = self.last_targets[leg_index,1] + alpha * (target_pos_y - self.last_targets[leg_index,1])

            if target_pos_z > self.last_targets[leg_index,2]:
                alpha = alpha_up*self.hysteresis_randomization[0]
            else:
                alpha = alpha_down*self.hysteresis_randomization[1]
            thz = self.last_targets[leg_index,2] + alpha * (target_pos_z - self.last_targets[leg_index,2])
                                                            
                #print(f"Leg {leg_index}, Joint {joint_index}: {joint_name}")
            # Apply to this leg's revolute joints
            for j in range(revolutes_per_leg-2):  
                joint_index = self.revolute_joints[leg_index * revolutes_per_leg + j]

                # Alternate X and Y for revolute joints
                if j % 2 == 0:
                    target, pg, vg = thx, pos_gain_x, vel_gain_x
                else:
                    target, pg, vg = thy, pos_gain_y, vel_gain_y

                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    positionGain=pg,
                    velocityGain=vg,
                    force=self.max_force,
                    maxVelocity=self.max_vel
                )
                #print(f"[REVOLUTE] Joint {joint_index} -> Target: {target:.3f}, PG: {pg:.2f}, VG: {vg:.2f}")
            # Apply to this leg's prismatic joints
            #total_pos_z = 0
            #total_target_pos_z = 0
            cup_link_idx = [leg_index*(self.no_links*4+2) + self.no_links*4 + 1]
            #print(cup_link_idx)
            for j in range(prismatics_per_leg):
                joint_index = self.prismatic_joints[leg_index * prismatics_per_leg + j]
                if joint_index not in cup_link_idx:                
                    #total_pos_z += p.getJointState(self.robot_id, joint_index)[0]
                    #total_target_pos_z += target_pos_z
                
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_index,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=thz,
                        positionGain=pos_gain_z,
                        velocityGain=vel_gain_z,
                        force=self.max_force,
                        maxVelocity=self.max_vel
                    )
            self.last_targets[leg_index,0] = thx
            self.last_targets[leg_index,1] = thy
            self.last_targets[leg_index,2] = thz
        self._current_pressures = np.array(pressures).flatten().tolist()  # Save for logging
                
    def apply_suction(self, suction_states, force_magnitude=10.0):
        #print(suction_states>50)
        """
        Simulate suction cups as prismatic actuators + contact/friction modifiers.
        suction_states: array-like of [0..100], suction % for each leg.
        """
        self.suction_states = suction_states
        angle_threshold = math.cos(math.radians(15))   # must be within 15° of vertical
        suction_pos_gain = 0.5                        # softer than default 0.1
        suction_vel_gain = 1.0                        # keep damping ~1
        suction_extension = 0.005                     # max prismatic stroke (5 mm)

        for leg in range(4):
            suction_state = suction_states[leg]
            cup_joint_idx = leg * (self.no_links * 4 + 2) + self.no_links * 4 + 1  
            #print("cup joint index = ", cup_joint_idx)
 
            target_extension = -(suction_state / 100.0) * suction_extension
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=cup_joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_extension,
                positionGain=suction_pos_gain,
                velocityGain=suction_vel_gain,
                force=self.max_force
            )
            # print extension
            current_extension = p.getJointState(self.robot_id, cup_joint_idx)[0]
            #print("target extension leg ", leg, " = ", target_extension, "current extension =", current_extension)
            # print max force for debugging
            

            # --- 2) Orientation & contact check ---
            pos_cup, orient_cup = p.getLinkState(self.robot_id, cup_joint_idx)[0:2]
            pos_plane = p.getBasePositionAndOrientation(self.plane_id)[0]

            # get z-axis of suction cup
            rot_matrix = p.getMatrixFromQuaternion(orient_cup)
            z_axis = rot_matrix[6:9]  
            alignment = sum(z_axis[i] * [0, 0, 1][i] for i in range(3))  

            # distance to plane
            dz = pos_plane[2] - pos_cup[2]
            distance = abs(dz)

            engaged = (
                distance < 0.02            # within 2 cm
                and alignment > angle_threshold  # close to vertical
                and suction_state > 10      # nonzero suction
            )

            # --- 3) Apply suction if engaged ---
            if engaged:
                # attract cup toward plane
                force = [0, 0, -force_magnitude * (suction_state/100.0) * self.suction_randomization]
                p.applyExternalForce(
                    objectUniqueId=self.robot_id,
                    linkIndex=cup_joint_idx ,
                    forceObj=force,
                    posObj=pos_cup,
                    flags=p.WORLD_FRAME
                )

                # lock cup with very high friction
                p.changeDynamics(
                    bodyUniqueId=self.robot_id,
                    linkIndex=cup_joint_idx + 1,
                    lateralFriction=5.0 + 0.1 * suction_state,
                    rollingFriction=0.01,
                    spinningFriction=0.01
                )
            else:
                # not engaged → friction back to normal
                p.changeDynamics(
                    bodyUniqueId=self.robot_id,
                    linkIndex=cup_joint_idx + 1,
                    lateralFriction=5.0,
                    rollingFriction=0.01,
                    spinningFriction=0.01
                )

        # --- 4) Logging pressures + suction ---
        if self._current_pressures is not None:
            log_row = self._current_pressures + suction_states.tolist()
            self.log_data.append(log_row)
            self._current_pressures = None


    def step(self):
        p.stepSimulation()
        if self.gui:
            time.sleep(1. / 240.)
                
    def run(self, steps=200, pressure_callback=None, suction_callback=None, mode=0):
        control_hz=24
        pressures = np.ones((4, 3)) * 10
        suction_states = np.array([0, 100, 100, 100], dtype=np.float64)
        
        control_period = 1.0 / control_hz

        start_time = time.time()

        for step in range(steps):
            step_start = time.time()

            # --- callbacks update pressures and suctions
            if pressure_callback:
                pressures += pressure_callback(step, mode)
            self.apply_pressures(pressures)

            if suction_callback:
                suction_states += suction_callback(step, mode)
            self.apply_suction(suction_states)
            # --- run multiple physics substeps per control cycle
            self.step()

            # --- enforce real-time control frequency
            if self.gui:
                elapsed = time.time() - step_start
                sleep_time = control_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Simulation ran for {steps} control steps in {duration:.2f} seconds.")
        print(f"Achieved control frequency: {steps/duration:.2f} Hz")

    def get_pressure_state(self):
        return self.pressure_state
    
    def get_suction_state(self):
        return self.suction_states

    def get_rotation_matrix(self):
        _, ori = p.getBasePositionAndOrientation(self.robot_id)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3,3)
        return rotation_matrix
    
    def get_base_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return pos
    
    def get_base_quaternion(self):
        _, ori = p.getBasePositionAndOrientation(self.robot_id)
        return ori

    def get_base_orientation(self):
        """
        Returns: np.array [roll, pitch, sin(yaw), cos(yaw)]
        """
        # Get quaternion from PyBullet
        _, quat = p.getBasePositionAndOrientation(self.robot_id)
        x, y, z, w = quat

        # Convert to roll, pitch, yaw
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        # Compute sin and cos of yaw
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)

        return np.array([roll, pitch, sin_yaw, cos_yaw], dtype=np.float32)
    
    def get_base_velocity(self):
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        return np.concatenate([lin_vel, ang_vel])
    
    def get_base_velocity_local(self):
        _, ori = p.getBasePositionAndOrientation(self.robot_id)
        rotation_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3,3)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        lin_vel = np.array(lin_vel) #to account for difference in pybullet and control frequency
        ang_vel = np.array(ang_vel) #to account for difference in pybullet and control frequency
        lin_vel_body = np.dot(rotation_matrix.T, lin_vel)
        ang_vel_body = np.dot(rotation_matrix.T, ang_vel)
        return np.concatenate([lin_vel_body, ang_vel_body])
    
    def get_grfs(self):
        """
        Returns a list of 12 values: the x,y,z ground reaction forces for each of the 4 feet.
        If a foot is not in contact, its forces will be [0, 0, 0].
        """
        grfs = np.zeros((4, 3))  # 4 feet, 3 force components (x, y, z)

        robot_id = self.robot_id
        foot_link_indices = self.foot_link_indices

        # Query all contact points involving the robot
        contact_points = p.getContactPoints(bodyA=robot_id)

        for contact in contact_points:
            link_idx = contact[3]  # linkIndexA
            if link_idx in foot_link_indices:
                i = np.where(foot_link_indices == link_idx)[0][0]                
                normal_force = contact[9] * np.array(contact[7])  # magnitude * normal vector
                lateral1 = contact[10] * np.array(contact[11])
                lateral2 = contact[12] * np.array(contact[13])
                total_force = normal_force + lateral1 + lateral2
                grfs[i] += total_force  # accumulate in case of multiple contacts
        return grfs.flatten()
    

    def get_grfs_local_foot(self):
        """
        Returns a list of 4 values: the z ground reaction forces for each of the 4 feet,
        converted to each foot's local frame.

        If a foot is not in contact, its force will be 0.
        """
        grfs_local = np.zeros(4)  # 4 feet

        robot_id = self.robot_id
        foot_link_indices = self.foot_link_indices

        # Query all contact points involving the robot
        contact_points = p.getContactPoints(bodyA=robot_id)

        for contact in contact_points:
            link_idx = contact[3]  # linkIndexA
            if link_idx in foot_link_indices:
                i = np.where(foot_link_indices == link_idx)[0][0]

                # Get the total contact force in world frame
                normal_force = contact[9] * np.array(contact[7])
                lateral1 = contact[10] * np.array(contact[11])
                lateral2 = contact[12] * np.array(contact[13])
                total_force_world = normal_force + lateral1 + lateral2

                # Get foot link orientation (world frame)
                _, foot_quat = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)[:2]

                # Rotation matrix from foot local to world
                rot_matrix = np.array(p.getMatrixFromQuaternion(foot_quat)).reshape(3, 3)

                # Rotate world force into foot local frame (inverse rotation)
                total_force_local = rot_matrix.T @ total_force_world

                grfs_local[i] += total_force_local[2]  # accumulate if multiple contacts
        #print(grfs_local)

        return grfs_local.flatten()

    def disconnect(self):
        # Save log to CSV
        if self.data_logging:
            if self.log_data:
                import csv
                header = [f"p_{i}" for i in range(12)] + [f"suction_{i}" for i in range(4)]
                with open("rl/log/tasq_log.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(self.log_data)
                print("[INFO] Pressure and suction data saved to tasq_log.csv")
        #p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
        p.disconnect()

    





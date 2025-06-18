import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import logging
from rl.converters.converter import pneu2posandgains

class TASQSimulator:
    def __init__(self, urdf_path, gui=False, no_links=0, render_mode=None):
        self.start_yaw = np.random.uniform(-np.pi, np.pi)
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
        p.setGravity(0, 0, -9.8)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                 cameraYaw=90+self.start_yaw*(180/np.pi),
                                 cameraPitch=-20,
                                 cameraTargetPosition=[0, 0, 0.2])
        
        self._load_environment()        
        self._initialize_robot()

        self.data_logging = False
        self.log_data = []  # Buffer for logging pressure + suction
        self._current_pressures = None  # Temporarily holds last pressures

        # Create a logger
        self.logger = logging.getLogger("suction_logger")
        self.logger.setLevel(logging.WARNING)  # Only log WARNING and above

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Create formatter and attach to handler
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)

        # Attach handler to logger
        self.logger.addHandler(console_handler)
    
    def _load_environment(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.start_x = np.random.uniform(-0.2, 0.2)
        self.start_y = np.random.uniform(-0.2, 0.2)
        self.start_pos = [self.start_x, self.start_y, 0.12]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, self.start_yaw])
        self.robot_id = p.loadURDF(self.urdf_path, self.start_pos, self.start_orientation)
        """
        print(f"[DEBUG] Robot loaded with ID: {self.robot_id}")
        num_joints = p.getNumJoints(self.robot_id)
        print(f"[DEBUG] Number of joints in robot: {num_joints}")
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            print(f"Joint {i}: name={joint_info[1].decode()}, type={joint_info[2]}")
        """
            
    def _initialize_robot(self):
        self.revolute_joints = []
        self.prismatic_joints = []
        self.suction_states = np.array([False, True, True, True])
        self.gait_length = 2300
        self.cycle_time = 0

        self.num_joints = p.getNumJoints(self.robot_id)

        self.position_gain = 1.0
        self.velocity_gain = 0.5
        self.max_force = 500

        excluded_names = {"j1_ps", "j2_ps", "j3_ps", "j4_ps"}

        for joint_index in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode()
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.revolute_joints.append(joint_index)
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
                force=self.max_force
            )

        # Foot link indices
        self.foot_link_indices = np.zeros(4)
        for leg_idx in range(4):
            cup_link_idx = leg_idx*(self.no_links*4+2) + self.no_links*4 +1
            self.foot_link_indices[leg_idx] = cup_link_idx  # list of 4 ints


    def apply_pressures(self, pressures):
        #print(pressures)
        
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
                #if leg_index==2:
                    #print("gain x = ",pos_gain_x, "gain y =", pos_gain_y, "gain z = ", pos_gain_z)            
            else:
                self.logger.warning(f"Pressure in leg  {leg_index +1} exceeding max_pressure of {max_pressure} mBar")

            # Apply to this leg's revolute joints
            for j in range(revolutes_per_leg):
                joint_index = self.revolute_joints[leg_index * revolutes_per_leg + j]
                
                # Alternate X and Y for revolute joints
                if j % 2 == 0:
                    target, pg, vg = target_pos_x, pos_gain_x, vel_gain_x
                else:
                    target, pg, vg = target_pos_y, pos_gain_y, vel_gain_y

                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    positionGain=pg,
                    velocityGain=vg,
                    force=self.max_force
                )
                #print(f"[REVOLUTE] Joint {joint_index} -> Target: {target:.3f}, PG: {pg:.2f}, VG: {vg:.2f}")
            # Apply to this leg's prismatic joints
            total_pos_z = 0
            total_target_pos_z = 0
            for j in range(prismatics_per_leg):
                joint_index = self.prismatic_joints[leg_index * prismatics_per_leg + j]                
                total_pos_z += p.getJointState(self.robot_id, joint_index)[0]
                total_target_pos_z += target_pos_z
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos_z,
                    positionGain=pos_gain_z,
                    velocityGain=vel_gain_z,
                    force=self.max_force
                )
                #print(f"[PRISMATIC] Joint {joint_index} -> Target: {target_pos_z:.3f}, PG: {pos_gain_z:.2f}, VG: {vel_gain_z:.2f}")
            if leg_index==1:
                1
                #print("total pos z = ", total_pos_z, "total target_pos_z =", total_target_pos_z)

        self._current_pressures = np.array(pressures).flatten().tolist()  # Save for logging
                
    def apply_suction(self, suction_states, force_magnitude=3):
        self.suction_states = suction_states
        angle_threshold = math.cos(math.radians(30))  # Cup must be within 15° of vertical
        suction_target_pos = -0.005
        suction_pos_gain = 0.8
        suction_vel_gain = 0.5
        no_suction_pos_gain = 0.5
        no_suction_vel_gain = 0.5
        for leg in range(4):
            suction_state = suction_states[leg]
            if suction_state:                
                cup_link_idx = leg*(self.no_links*4+2) + self.no_links*4                
                                
                # Get positions
                pos_cup, orient_cup = p.getLinkState(self.robot_id, cup_link_idx)[0:2]
                pos_plane = p.getBasePositionAndOrientation(self.plane_id)[0]

                # Orientation constraint
                rot_matrix = p.getMatrixFromQuaternion(orient_cup)
                z_axis = rot_matrix[6:9]
                angle_error = sum(z_axis[i] * [0, 0, 1][i] for i in range(3))  # Dot product with world Z
                                
                # Compute direction and apply force
                direction = [0,0,pos_plane[2] - pos_cup[2]]
                distance = sum(d ** 2 for d in direction) ** 0.5
                if distance > 0 and distance < 0.03 and angle_error > angle_threshold:
                    #"""Apply an attracting force from bodyB:linkB to self.robot_id:cup_link_idx."""
                    force = [0,0,-force_magnitude]  
                    p.applyExternalForce(
                        objectUniqueId=self.robot_id,
                        linkIndex=cup_link_idx,
                        forceObj=force,
                        posObj=pos_cup,
                        flags=p.WORLD_FRAME
                    )
                    # Change the friction force between the suction cup and the ground plane
                    p.changeDynamics(
                        bodyUniqueId=self.robot_id,
                        linkIndex=cup_link_idx+1,             # Use -1 for base link
                        lateralFriction=5       # New friction value
                    )
                    #Change the suction cup target_position
                #"""
                    p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=cup_link_idx+1,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=suction_target_pos,
                    positionGain=suction_pos_gain,
                    velocityGain=suction_vel_gain,
                    force=self.max_force                    
                    )
                    
                else:
                    p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=cup_link_idx+1,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0,
                    positionGain=no_suction_pos_gain,
                    velocityGain=no_suction_vel_gain,
                    force=self.max_force
                )
                    p.changeDynamics(
                        bodyUniqueId=self.robot_id,
                        linkIndex=cup_link_idx,             # Use -1 for base link
                        lateralFriction=0.5    # New friction value
                    )
                #"""
        """
        if np.array_equal(suction_states, self.prev_suction_states):
            self.suction_states = self.prev_suction_states

        else:
            self.step_count += 1
            self.suction_states = suction_states
            self.prev_suction_states = suction_states

        if self.step_count >= 4:
            self.cycle_time = 0
            self.step_count = 0
        """
        #pressue and suction logging
        if self._current_pressures is not None:
            log_row = self._current_pressures + suction_states.tolist()
            self.log_data.append(log_row)
            self._current_pressures = None  # Reset after logging
    
    def change_gait_length(self, new_gait_length):
        self.gait_length = new_gait_length

    def step(self):
        self.cycle_time += 1
        if self.cycle_time > self.gait_length:
            self.cycle_time = 0
        p.stepSimulation()
        if self.gui:
            time.sleep(1. / 240.)
                
    def run(self, steps=2000, pressure_callback=None, suction_callback=None):
        pressures = np.zeros((4, 3))
        suction_states = np.array([False, False, False, False])
        #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./tasq_rl/videos/simulation_record.mp4")

        for step in range(steps):            
            if pressure_callback:
                pressures += pressure_callback(step)
                self.apply_pressures(pressures)

            if suction_callback:
                suction_states = suction_callback(step)
                self.apply_suction(suction_states)            
            self.step()

    def get_suction_state(self):
        return self.suction_states

    def get_time(self):
        return np.array([self.cycle_time])
    
    def get_gait_length(self):
        return self.gait_length
    
    def get_base_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return pos
    
    def get_base_orientation(self):
        _, ori = p.getBasePositionAndOrientation(self.robot_id)
        return ori
    
    def get_base_velocity(self):
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        return np.concatenate([lin_vel, ang_vel])
    
    def get_grfs(self):
        """
        Returns a list of 12 values: the x, y, z ground reaction forces for each of the 4 feet.
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

    





import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
from rl.scripts.tasq_simulator import TASQSimulator  # Adjust this import path as needed
from collections import deque
import torch
import torch.nn.functional as F
import time

OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base_angular_velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:35,  # phase
    35:39   # goal_pos and orientation
]

def rotate_world_to_local(base_pos, base_quat, point):
    vec = np.array([point[0] - base_pos[0], point[1] - base_pos[1], 0.0])
    rot_matrix = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
    local_vec = rot_matrix.T @ vec
    return local_vec[0:2]

class TASQEnv(gym.Env):
    def __init__(self, urdf_path, no_links, gui=False, render_mode=None, frame_stack=4):
        super().__init__()
        self.urdf_path = urdf_path
        self.no_links = no_links
        self.sim = TASQSimulator(self.urdf_path, gui=gui, no_links=self.no_links, render_mode=render_mode)
        self.max_steps = 12*5
        self.step_count = 0
        self.previous_step_time = time.time()
        self.gait_length = 12

        # --- Frame stacking ---
        self.frame_stack = frame_stack
        self.obs_buffer = deque(maxlen=frame_stack)
        
        # Initialize pressures and suction states
        self.pressures = np.ones((4,3), dtype=np.float32)*10
        self.pressures[1,:] = 400
        self.pressures[3,:] = 400
        self.suction_states = np.array([0,0,100,0], dtype=np.float32)


        # Initialize deltas
        self.d_pressures = np.zeros((4,3), dtype=np.float32)
        self.d_suctions = np.zeros((4,), dtype=np.float32)
           
        # Observation space: Stacked observations
        self.obs_dim = 33
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim * frame_stack,), dtype=np.float32
        )

        # Actions = [12 pressure deltas + 4 suction states]
        self.action_dim = 12 + 4
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )



        self.target_pos = np.array([0.3, 0.0], dtype=np.float32) # Target position for the robot (used in reward calculation)
        self.target_ori = np.array([np.sin(np.pi/4), np.cos(np.pi/4)], dtype=np.float32) # Target orientation for the robot (used in reward calculation)
        self.target_height = 0.12  # Set a target height for the robot
        self.goal_pos = np.array([0.01, 0.0], dtype=np.float32) # Relative goal position for the robot (commanded goal direction) 
        self.grf_buffer = []
        self.vel_buffer = []
        self.buffer_len = 100
        self.grf_randomizer = np.random.normal(0, 3, size=(4,))
        self.vel_randomizer = np.random.normal(0, 0.01, size=(6,))
        self._np_random = None
        self.seed()  # initialize RNG

    def seed(self, seed: int = None):
        """Set the random seed for reproducibility."""
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset_simulation()
        self.step_count = 0
        self.total_x_reward = 0.0
        self.last_suctions = np.zeros(4, dtype=np.float32)
        self.target_height = 0.12  # Set a target height for the robot
        self.target_pos = np.array([0.3, 0.0], dtype=np.float32)  # Reset target position
        self.target_ori = np.array([np.sin(np.pi/4), np.cos(np.pi/4)], dtype=np.float32)  # Reset target orientation
        self.goal_pos = np.array([0.01, 0.0], dtype=np.float32)  # Reset goal position
        self.goal_yaw = np.pi/4
        #for i in range(100):
        #    self.pressures[1,:] += 3.9
        #    self.pressures[3,:] += 3.9
        #    self.sim.apply_pressures(self.pressures)
        #    self.sim.apply_suction(self.suction_states)
        #    self.sim.step()
        obs = self._get_obs()
        self.obs_buffer.clear()
        for _ in range(self.frame_stack):
            self.obs_buffer.append(obs)
        return np.concatenate(self.obs_buffer), {}


    def _get_obs(self):
        state = self._get_state()
        # Select only the observed indices
        obs = state[OBSERVED_INDICES].astype(np.float32)
        obs[0:4] *= np.random.normal(1, 0.01, size=(4,))  # Add noise to orientation      

        return obs
    
    def _get_state(self):
        base_pos = self.sim.get_base_position()         
        base_ori = self.sim.get_base_orientation()
        base_vel = self.get_mean_vel()  

        #pressures and suctions
        normalized_pressures = (self.pressures / 400).flatten()
        normalized_suction = self.suction_states/100

        #grfs
        grfs = self.get_mean_grf()
        normalized_grfs = grfs / 30   

        #phase
        raw_phase = ((self.step_count % self.gait_length) / self.gait_length) * 2 * np.pi
        phase = np.array([np.sin(raw_phase), np.cos(raw_phase)])
        
        #goal pos and orientation
        goal_local_xy = rotate_world_to_local(base_pos[0:2], base_ori, self.goal_pos[0:2])
        base_yaw = np.arctan2(base_ori[2], base_ori[3])
        delta_yaw = self.goal_yaw - base_yaw
        delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
        sin_delta_yaw = np.sin(delta_yaw)
        cos_delta_yaw = np.cos(delta_yaw)
        goal_ori = np.array([sin_delta_yaw, cos_delta_yaw])
        goal_ori = np.array([0,1])


        # Combine all into one observation array
        obs = np.concatenate([
            base_pos,               # shape (3,)               
            base_ori,               # shape (4,)
            base_vel,               # shape (6,)
            normalized_pressures,   # shape (12,)
            normalized_suction,     # shape (4,)
            normalized_grfs,        # shape (4,)
            phase,                  # shape (2,)
            goal_local_xy,          # shape (2,)
            goal_ori           # shape (2,)
        ]).astype(np.float32)

        return obs
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        self.d_pressures = (action[:12].reshape(4, 3)) * 400
        self.d_suctions = (action[12:16]) * 100  

        # Initialize x history if not present
        if not hasattr(self, 'x_history'):
            self.x_history = deque(maxlen=2)
        self.x_history.append(self.sim.get_base_position()[0])

        for i in range(100):
            grfs = self.sim.get_grfs_local_foot()
            vel = self.sim.get_base_velocity_local()
            self.update_buffers(grfs, vel)

            # Apply pressure deltas
            self.pressures = np.clip(self.pressures + self.d_pressures/100, 0.0, 400.0)
            self.suction_states = np.clip(self.suction_states + self.d_suctions/100, 0, 100)
            self.sim.apply_pressures(self.pressures)
            self.sim.apply_suction(self.suction_states)
            self.sim.step()

        obs = self._get_obs()
        self.obs_buffer.append(obs)
        stacked_obs = np.concatenate(self.obs_buffer)

        state = self._get_state()
        current_height = state[2]
        roll, pitch, sin_yaw, cos_yaw = state[3:7]

        # ------------------- Reward Weights -------------------
        weights = {
            "alive": 0.02,
            "forward": 20.0,
            "orientation_penalty": 0.1,
            "height_penalty": 0.1,
            "pressure_smoothness": 1e-8,
            "suction_smoothness": 1e-6
        }

        # ------------------- Reward Components -------------------
        reward_terms = {}

        reward_terms["alive"] = weights["alive"] * 1.0

        # Forward x-motion over last 2 steps
        if len(self.x_history) == 2:
            reward_terms["forward"] = weights["forward"] * (self.x_history[-1] - self.x_history[0])
        else:
            reward_terms["forward"] = 0.0

        # Orientation penalty
        reward_terms["orientation_penalty"] = -weights["orientation_penalty"] * (np.abs(roll) + np.abs(pitch))

        # Height penalty
        reward_terms["height_penalty"] = -weights["height_penalty"] * np.abs(current_height - self.target_height)

        # Action smoothness penalties
        reward_terms["pressure_smoothness"] = -weights["pressure_smoothness"] * np.sum(np.square(self.d_pressures))
        reward_terms["suction_smoothness"] = -weights["suction_smoothness"] * np.sum(np.square(np.abs(self.d_suctions)))

        # Total step reward
        reward = sum(reward_terms.values())
        #if self.step_count % 10 == 0:
            #print("total reward:", reward)
        reward = np.clip(reward, -1.0, 1.0)

        # ------------------- Termination -------------------
        terminated = current_height < 0.05
        truncated = self.step_count >= self.max_steps
        self.step_count += 1

        # Accumulate episodic reward for total x-motion
        if truncated or terminated:
            self.total_x_reward = self.sim.get_base_position()[0]

        info = {
            "reward_terms": reward_terms,
            "total_x_reward": self.total_x_reward
        }

        return stacked_obs, reward, terminated, truncated, info


    def render(self):
        return None

    def close(self):
        self.sim.disconnect()

    def update_buffers(self, grfs, base_vel):
        self.grf_buffer.append(grfs)
        if len(self.grf_buffer) > self.buffer_len:
            self.grf_buffer.pop(0)
        self.vel_buffer.append(base_vel)
        if len(self.vel_buffer) > self.buffer_len:
            self.vel_buffer.pop(0)

    def get_mean_grf(self):
        if len(self.grf_buffer) == 0:
            return np.zeros(4)
        mean_grfs = np.mean(self.grf_buffer, axis=0) + self.grf_randomizer
        return mean_grfs

    def get_mean_vel(self):
        if len(self.vel_buffer) == 0:
            return np.zeros(6)
        mean_vel = np.mean(self.vel_buffer, axis=0) + self.vel_randomizer
        return mean_vel

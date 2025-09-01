import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
from rl.scripts.tasq_simulator import TASQSimulator  # Adjust this import path as needed

OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base_angular_velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:37   # goal_pos and orientation
]

class TASQEnv(gym.Env):
    def __init__(self, urdf_path, no_links, gui=False, render_mode=None, frame_stack=4):
        super().__init__()
        self.urdf_path = urdf_path
        self.no_links = no_links
        self.sim = TASQSimulator(self.urdf_path, gui=gui, no_links=self.no_links, render_mode=render_mode)
        self.max_steps = 20000
        self.step_count = 0

        # --- Frame stacking ---
        self.frame_stack = frame_stack
        self.obs_buffer = deque(maxlen=frame_stack)
        
        # Initialize pressures and suction states
        self.pressures = np.ones((4,3)*10, dtype=np.float32)
        self.suction_states = np.arrary(([0,100,100,100]), dtype=np.float32)
           
        # Observation space: Stacked observations
        self.obs_dim = len(self._get_state())
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
        self.goal_pos = np.array([0.3, 0.0], dtype=np.float32) # Relative goal position for the robot (commanded goal direction) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset_simulation()
        self.step_count = 0
        self.last_suctions = np.zeros(4, dtype=np.float32)
        self.target_height = 0.12  # Set a target height for the robot
        self.target_pos = np.array([0.3, 0.0], dtype=np.float32)  # Reset target position
        self.target_ori = np.array([np.sin(np.pi/4), np.cos(np.pi/4)], dtype=np.float32)  # Reset target orientation
        self.goal_pos = np.array([0.3, 0.0], dtype=np.float32)  # Reset goal position
        self.goal_ori = np.array([np.sin(0.0), np.cos(0,0)], dtype=np.float32)  
        obs = self._get_obs()
        self.obs_buffer.clear()
        for _ in range(self.frame_stack):
            self.obs_buffer.append(obs)
        return np.concatenate(self.obs_buffer), {}


    def _get_obs(self):
        state = self._get_state()
        # Select only the observed indices
        obs = state[OBSERVED_INDICES].astype(np.float32)

        return obs
    
    def _get_state(self):
        base_pos = self.sim.get_base_position()         
        base_ori = self.sim.get_base_orientation()      
        base_vel = self.sim.get_base_velocity_local()   
        normalized_pressures = (self.pressures / 200 - 1).flatten()
        normalized_suction = self.suction_states/50 -1
        grfs = self.sim.get_grfs_local_foot()         
        normalized_grfs = grfs / 10

        # Combine all into one observation array
        obs = np.concatenate([
            base_pos,               # shape (3,)               
            base_ori,               # shape (4,)
            base_vel,               # shape (6,)
            normalized_pressures,   # shape (12,)
            normalized_suction,     # shape (4,)
            normalized_grfs,        # shape (4,)
            self.goal_pos,          # shape (2,)
            self.goal_ori           # shape (2,)
        ]).astype(np.float32)

        return obs
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        d_pressures = (action[:12].reshape(4, 3)) * 20 * np.pi
        d_suctions = (action[12:16]) * 20  

        obs = self._get_obs()
        self.obs_buffer.append(obs)
        stacked_obs = np.concatenate(self.obs_buffer)

        control_frequency = 1./24.
        pybullet_frequency = 1./240.
        steps_per_step = int(control_frequency / pybullet_frequency)

        for i in range(steps_per_step):
            # Apply pressure deltas
            self.pressures = np.clip(self.pressures + d_pressures/steps_per_step, 0.0, 400.0)
            self.suction_states = np.clip(self.suction_states + d_suctions/steps_per_step, 0, 100)
            # Step the simulator with current pressure and suction state
            self.sim.apply_pressures(self.pressures)
            self.sim.apply_suction(self.suction_states)
            #print(self.pressures, self.suction_states)
            self.sim.step()

        # Calculate reward
        state = self._get_state()

        # 1. Target reward
        pos = state[:2]
        distance = np.linalg.norm(pos - self.target_pos)
        roll, pitch, sin_yaw, cos_yaw = state[3:7]
        relative_ori = np.array([sin_yaw, cos_yaw]) - self.target_ori
        orientation_reward = np.linalg.norm(relative_ori)
        
        # 2. Height deviation penalty
        current_height = state[2]
        height_penalty = np.abs(current_height - self.target_height)

        # 3. Orientation deviation penalty 
        orientation_penalty = np.abs(pitch) + np.abs(roll)

        # 4. Smoothness penalty (based on pressure change)
        pressure_smoothness_penalty = np.sum(np.square(d_pressures))

        # 4. Smoothness penalty (based on suction change)
        suction_smoothness_penalty = np.sum(np.square(np.abs(d_suctions)))

        # --- Weighting factors ---
        w_dist = 1.0
        w_ori = 1.0
        w_height = 2.0
        w_orient = 2.0
        w_pressure_smoothness = 0.1
        w_suction_smoothness = 0.1
        #w_gait_length

        reward_terms = {
            "distance": -w_dist * distance,
            "orientation_reward": -w_ori * orientation_reward,
            "height_penalty": -w_height * height_penalty,
            "orientation_penalty": -w_orient * orientation_penalty,
            "pressure_smoothness": -w_pressure_smoothness * pressure_smoothness_penalty,
            "suction_smoothness": -w_suction_smoothness * suction_smoothness_penalty,
        }

        reward = sum(reward_terms.values())

        # Terminate or truncate
        terminated = distance < 0.0005
        truncated = self.step_count >= self.max_steps

        self.step_count += 1
        return stacked_obs, reward, terminated, truncated, {"reward_terms": reward_terms}

    def render(self):
        return None

    def close(self):
        self.sim.disconnect()

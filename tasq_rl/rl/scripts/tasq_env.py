import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
from rl.scripts.tasq_simulator import TASQSimulator  # Adjust this import path as needed

class TASQEnv(gym.Env):
    def __init__(self, urdf_path, no_links, gui=False, render_mode=None):
        super().__init__()
        self.urdf_path = urdf_path
        self.no_links = no_links
        self.sim = TASQSimulator(self.urdf_path, gui=gui, no_links=self.no_links, render_mode=render_mode)
        self.max_steps = 20000
        self.step_count = 0

        # Initialize pressures and suction states
        self.pressures = np.zeros((4,3), dtype=np.float32)
        self.suction_states = np.zeros(4, dtype=np.uint8)
           
        # Action = 12 pressure deltas in [-1, 1] + 4 suction cup states (binary 0 or 1)
        self.action_space = spaces.Box(
            low=np.array([-1.0] * 12 + [0.0] * 4, dtype=np.float32),
            high=np.array([1.0] * 12 + [1.0] * 4, dtype=np.float32),
            dtype=np.float32
        )

        # Observation: base pos (3,), orientation (4,), velocities(6,), pressures (12,), suction states (4,), grfs (12,)
        # For now: base x, y, z, and orientation (quaternion) and velocities
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32)

        self.target = np.array([0.0, 0.0], dtype=np.float32)  # You can randomize this in `reset`

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Instead of disconnect/recreate sim, just reset sim state:
        self.sim.reset_simulation()  # you may need to add this method in TASQSimulator
        self.step_count = 0
        self.target = np.array([0.0, 0.0], dtype=np.float32)        
        obs = self._get_obs()
        return obs, {}


    def _get_obs(self):
        # Get base state
        base_pos = self.sim.get_base_position()          # shape (3,)
        base_quat = self.sim.get_base_orientation()      # shape (4,)
        base_vel = self.sim.get_base_velocity()          # shape (6,)

        # Pressures (already tracked)
        pressures = self.pressures.flatten()/200 -1                             # shape (12,)

        # Suction cup states (binary)
        suction = self.suction_states.astype(np.float32)       # shape (4,)

        # Ground reaction forces
        grfs = self.sim.get_grfs()                       # shape (12,)

        # Combine all into one observation array
        obs = np.concatenate([
            base_pos,
            base_quat,
            base_vel,
            pressures,
            suction,
            grfs
        ]).astype(np.float32)

        return obs
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        d_pressures = (action[:12].reshape(4, 3)) * 2 * np.pi
        suction_raw = action[12:16]
        #suction_sigmoids = 1/(1 + np.exp(-suction_raw))
        #print(suction_raw, suction_sigmoids)

        # Apply pressure deltas
        self.pressures = np.clip(self.pressures + d_pressures, 0.0, 400.0)

        # Update suction states based on threshold
        self.suction_states = (suction_raw > 0.5).astype(np.uint8)

        # Step the simulator with current pressure and suction state
        self.sim.apply_pressures(self.pressures)
        self.sim.apply_suction(self.suction_states)
        #print(self.pressures, self.suction_states)
        self.sim.step()

        # Calculate reward
        obs = self._get_obs()

        # 1. Position and target reward
        pos = obs[:2]
        distance = np.linalg.norm(pos - self.target)
        
        # 2. Height deviation penalty
        current_height = obs[2]
        target_height = self.target_height if hasattr(self, 'target_height') else 0.12  # adjust target height
        height_penalty = np.abs(current_height - target_height)

        # 3. Orientation deviation penalty (assume quaternion in obs[3:7])
        quat = obs[3:7]
        _, pitch, roll = p.getEulerFromQuaternion(quat)[0:3]  # or use scipy if preferred
        orientation_penalty = np.abs(pitch) + np.abs(roll)

        # 4. Smoothness penalty (based on pressure change)
        smoothness_penalty = np.sum(np.square(d_pressures))

        # --- Weighting factors ---
        w_dist = 1.0
        w_height = 2.0
        w_orient = 2.0
        w_smooth = 0.1
        #w_gait_length

        # Combine total reward
        reward = -(
            w_dist * distance +
            w_height * height_penalty +
            w_orient * orientation_penalty +
            w_smooth * smoothness_penalty
        )

        # Terminate or truncate
        terminated = distance < 0.0005
        truncated = self.step_count >= self.max_steps

        self.step_count += 1
        return obs, reward, terminated, truncated, {}

    def render(self):
        return None

    def close(self):
        self.sim.disconnect()

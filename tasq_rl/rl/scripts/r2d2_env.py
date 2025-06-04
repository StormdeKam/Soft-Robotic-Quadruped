import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time

class R2D2Env(gym.Env):
    def __init__(self, gui=False, render_mode=None):
        self.render_mode = render_mode  # Not used, but keeps Gymnasium happy
        super(R2D2Env, self).__init__()
        self.gui = gui
        # Ensure no existing connection
        try:
            p.disconnect()
        except Exception:
            pass
        self._connect()

        # Action: left and right wheel velocities
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation: x, y, vx, vy, target_x, target_y
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.target_pos = np.array([2.0, 2.0], dtype=np.float32)

    def _connect(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.1])
        self.steps = 0
        self.target_pos = np.random.uniform(-3, 3, size=2).astype(np.float32)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, _ = p.getBaseVelocity(self.robot_id)
        return np.array([pos[0], pos[1], lin_vel[0], lin_vel[1], self.target_pos[0], self.target_pos[1]], dtype=np.float32)

    def step(self, action):
        max_force = 10
        velocity_scaling = 10.0
        action = np.clip(action, -1, 1) * velocity_scaling

        # Apply wheel velocities
        p.setJointMotorControl2(
            self.robot_id, 2, p.VELOCITY_CONTROL,
            targetVelocity=action[0], force=max_force
        )
        p.setJointMotorControl2(
            self.robot_id, 3, p.VELOCITY_CONTROL,
            targetVelocity=action[1], force=max_force
        )

        p.stepSimulation()
        if self.gui:
            time.sleep(1 / 240)

        obs = self._get_obs()
        pos = obs[:2]
        distance = np.linalg.norm(pos - self.target_pos)

        reward = -distance
        terminated = bool(distance < 0.2)
        truncated = bool(self.steps > 1000)
        self.steps += 1

        return obs, reward, terminated, truncated, {}

    def render(self):
        # GUI rendering handled by PyBullet
        return None

    def close(self):
        p.disconnect()


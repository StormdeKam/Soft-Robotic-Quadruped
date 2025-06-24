from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_interval=10, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = None
        self.current_lengths = None

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.current_rewards = np.zeros(n_envs)
        self.current_lengths = np.zeros(n_envs)

    def _on_step(self):
        # Update rewards and lengths
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]
        self.current_rewards += rewards
        self.current_lengths += 1

        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                if self.verbose > 0:
                    print(f"[RewardLogger] Episode done: reward={self.current_rewards[i]:.3f}, length={self.current_lengths[i]}")
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0
        # Log at intervals
        #print(self.log_interval)
        if self.n_calls % self.log_interval == 0:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"[RewardLogger] Step {self.n_calls}: mean reward (last 100 eps): {mean_reward:.3f}")
        return True

    def _on_training_end(self):
        df = pd.DataFrame({
            "reward": self.episode_rewards,
            "length": self.episode_lengths
        })
        df.to_csv("reward_log.csv", index=False)
        print("[RewardLogger] Saved reward log to reward_log.csv")


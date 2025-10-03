from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import os

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, log_interval=10, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.log_path = log_path

        self.current_rewards = None
        self.current_lengths = None
        self.current_components = None

        # Ensure directory exists
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        # Create new log file with header if it doesn’t exist
        if not os.path.isfile(log_path):
            df = pd.DataFrame(columns=["reward", "length"])  
            df.to_csv(log_path, index=False)
            if verbose > 0:
                print(f"[RewardLogger] Created new log file at {log_path}")

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.current_rewards = np.zeros(n_envs)
        self.current_lengths = np.zeros(n_envs)
        self.current_components = [dict() for _ in range(n_envs)]

    def _on_step(self):
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]

        self.current_rewards += rewards
        self.current_lengths += 1

        # accumulate component rewards
        for i, info in enumerate(infos):
            if "reward_terms" in info:
                for k, v in info["reward_terms"].items():
                    self.current_components[i][k] = self.current_components[i].get(k, 0.0) + v

        # handle episode end
        for i, done in enumerate(dones):
            if done:
                row = {"reward": self.current_rewards[i], "length": self.current_lengths[i]}
                row.update(self.current_components[i])  # flatten reward_terms too

                # append immediately to file
                pd.DataFrame([row]).to_csv(self.log_path, mode="a", header=not os.path.isfile(self.log_path), index=False)

                if self.verbose > 0:
                    comp_str = ", ".join(f"{k}={v:.3f}" for k, v in self.current_components[i].items())
                    print(f"[RewardLogger] Episode done: total={self.current_rewards[i]:.3f}, {comp_str}")

                # reset trackers for that env
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0
                self.current_components[i] = dict()

        if self.n_calls % self.log_interval == 0:
            try:
                df = pd.read_csv(self.log_path)
                if not df.empty:
                    mean_reward = np.mean(df["reward"].tail(100))
                    print(f"[RewardLogger] Step {self.n_calls}: mean reward (last 100 eps): {mean_reward:.3f}")
            except Exception:
                pass

        return True

    def _on_training_end(self):
        print(f"[RewardLogger] Training finished, logs saved to {self.log_path}")

from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_interval=10, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_components = []  # store per-episode reward components

        self.current_rewards = None
        self.current_lengths = None
        self.current_components = None

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
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                self.episode_components.append(self.current_components[i])

                if self.verbose > 0:
                    print(f"[RewardLogger] Episode done: total={self.current_rewards[i]:.3f}, " +
                          ", ".join(f"{k}={v:.3f}" for k, v in self.current_components[i].items()))

                # reset trackers for that env
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0
                self.current_components[i] = dict()

        if self.n_calls % self.log_interval == 0 and self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])
            print(f"[RewardLogger] Step {self.n_calls}: mean reward (last 100 eps): {mean_reward:.3f}")

        return True

    def _on_training_end(self):
        # turn component dicts into dataframe
        df = pd.DataFrame(self.episode_components)
        df["reward"] = self.episode_rewards
        df["length"] = self.episode_lengths
        df.to_csv("reward_log.csv", index=False)
        print("[RewardLogger] Saved reward log to reward_log.csv")

from stable_baselines3.common.callbacks import BaseCallback
import time

class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, log_interval=10_000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.verbose:
            print("[Callback] Training started.")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            steps_done = self.num_timesteps
            steps_remaining = self.total_timesteps - steps_done
            steps_per_sec = steps_done / max(elapsed, 1e-6)
            eta = steps_remaining / max(steps_per_sec, 1e-6)

            hrs = int(eta // 3600)
            mins = int((eta % 3600) // 60)
            secs = int(eta % 60)

            if self.verbose:
                print(f"[Step {steps_done}/{self.total_timesteps}] "
                      f"Elapsed: {elapsed:.1f}s | ETA: {hrs}h {mins}m {secs}s")
        return True

from stable_baselines3 import SAC
from r2d2_env import R2D2Env

def test_model():
    env = R2D2Env(gui=True)
    model = SAC.load("../models/sac_r2d2")

    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    test_model()


import os
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from rl.scripts.tasq_env import TASQEnv
from rl.models.architecture.actor import BCActor

# ---------------- Config ----------------
SAVE_PATH = "datasets/replay_buffers/low_freq_v6"
os.makedirs(SAVE_PATH, exist_ok=True)

N_ENVS = 8
N_TRANSITIONS_PER_ENV = 1000
FRAME_STACK = 4
SEED = 42
BC_WEIGHTS_PATH = "rl/models/tasq_bc/low_freq_v6/tasq_bc_ep200.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NO_LINKS_LIST = [5, 6, 7]
URDF_BY_LINKS = {5: "rl/urdfs/TASQ_n5.urdf", 6: "rl/urdfs/TASQ_n6.urdf", 7: "rl/urdfs/TASQ_n7.urdf"}

# ---------------- Env Helpers ----------------
def make_env_fn(rank: int, seed: int = 0) -> callable:
    def _init():
        import numpy as np
        import torch
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        no_links = int(np.random.choice(NO_LINKS_LIST))
        urdf_path = URDF_BY_LINKS[no_links]
        env = TASQEnv(urdf_path=urdf_path, no_links=no_links, frame_stack=FRAME_STACK)
        env.seed(seed + rank)
        return env
    return _init

def make_vec_env(n_envs: int):
    env_fns = [make_env_fn(i, seed=SEED) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec)
    return vec

# ---------------- Main ----------------
def main():
    # Create parallel envs
    env = make_vec_env(N_ENVS)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load BC actor
    bc_actor = BCActor(obs_dim).to(DEVICE)
    state_dict = torch.load(BC_WEIGHTS_PATH, map_location=DEVICE)
    bc_actor.load_state_dict(state_dict)
    bc_actor.eval()

    # Storage
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []

    obs = env.reset()
    transitions_collected = 0

    while transitions_collected < N_TRANSITIONS_PER_ENV * N_ENVS:
        # Convert obs to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pressures, suctions = bc_actor(obs_tensor)
            action = torch.cat([pressures, suctions], dim=-1)
        action = action.cpu().numpy()
        
        # Step env
        next_obs, reward, done, info = env.step(action)
        
        # Store
        states.append(obs.copy())
        actions.append(action.copy())
        next_states.append(next_obs.copy())
        rewards.append(reward.copy())
        dones.append(done.copy())

        obs = next_obs
        transitions_collected += N_ENVS

        if transitions_collected % 1000 == 0:
            print(f"[INFO] Collected {transitions_collected} transitions")

    # Convert to arrays and save
    np.savez_compressed(
        os.path.join(SAVE_PATH, "replay_buffer3.npz"),
        states=np.array(states),
        actions=np.array(actions),
        next_states=np.array(next_states),
        rewards=np.array(rewards),
        dones=np.array(dones)
    )
    print(f"[DONE] Saved {transitions_collected} transitions to {SAVE_PATH}/replay_buffer3.npz")

if __name__ == "__main__":
    main()

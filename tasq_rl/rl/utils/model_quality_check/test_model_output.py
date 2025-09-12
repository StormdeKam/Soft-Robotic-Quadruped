import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tasq_rl.rl.models.architecture.actor import BCActor
from collections import defaultdict

# === Load model ===
MODEL_PATH = "rl/models/tasq_bc/goal_directed_4_8/tasq_bc_ep180_20250805-151622.pt"
DEVICE = "cpu"
OBS_DIM = 37 * 2 + 5 * 33

model = BCActor(obs_dim=OBS_DIM)
model.initialize_weights_from_checkpoint(MODEL_PATH, device=DEVICE)
model.eval()

# === Load data ===
demo_states = np.load("rl/datasets/goal_directed/demo_states_0.npy")
demo_actions = np.load("rl/datasets/goal_directed/demo_actions_0.npy")
demo_episode_lengths = np.load("rl/datasets/goal_directed/demo_episode_lengths_0.npy")

mse_loss_fn = nn.MSELoss()

start_index = 0
for ep_len in demo_episode_lengths:
    for t in range(ep_len):
        i = start_index + t

        full = demo_states[i]
        fullm1 = demo_states[i - 1] if t > 0 else full
        fullm2 = demo_states[i - 2] if t > 1 else full
        fullm3 = demo_states[i - 3] if t > 2 else full

        state = np.concatenate([full, fullm1, fullm2, fullm3])
        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pressures, suctions = model(input_tensor)
            predicted = torch.cat([pressures.squeeze(0), suctions.squeeze(0)])
            action = torch.tensor(demo_actions[i], dtype=torch.float32)
            loss = mse_loss_fn(predicted, action).item()


    start_index += ep_len
    print(f"Episode {i // ep_len} step {t}: Loss = {loss:.4f}")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rl.models.tasq_bc.bc_actor import BCActor
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

# === Indices for partial state ===
partial_idx = np.r_[
    0:3,    # base pos
    3:7,    # base orn
    7:13,   # base vel
    13:25,  # pressures
    25:29,  # suctions
    33:37   # goal_pos and orn
]

mse_loss_fn = nn.MSELoss()

# === Collect binned losses ===
binned_losses = defaultdict(list)

start_index = 0
for ep_len in demo_episode_lengths:
    for t in range(ep_len):
        i = start_index + t

        full = demo_states[i]
        fullm1 = demo_states[i - 1] if t > 0 else full

        # Recursive padding strategy
        def get_partial(offset):
            if t - offset >= 0:
                return demo_states[start_index + t - offset][partial_idx]
            elif offset == 200:
                return get_partial(160)
            elif offset == 160:
                return get_partial(120)
            elif offset == 120:
                return get_partial(80)
            elif offset == 80:
                return get_partial(40)
            else:  # offset == 40 or none available
                return demo_states[start_index][partial_idx]

        partials = [get_partial(off) for off in [200, 160, 120, 80, 40]]

        state = np.concatenate([full, fullm1] + partials)
        input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pressures, suctions = model(input_tensor)
            predicted = torch.cat([pressures.squeeze(0), suctions.squeeze(0)])
            action = torch.tensor(demo_actions[i], dtype=torch.float32)
            loss = mse_loss_fn(predicted, action).item()

        # Bin loss by timestep (every 10 steps)
        bin_idx = t // 10
        binned_losses[bin_idx].append(loss)

    start_index += ep_len

# === Compute average losses per bin ===
bins = sorted(binned_losses.keys())
avg_losses = [np.mean(binned_losses[b]) for b in bins]

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot([b * 10 for b in bins], avg_losses, marker='o')
plt.xlabel("Timestep")
plt.ylabel("Average MSE Loss")
plt.title("Binned MSE Loss over Timesteps")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/binned_mse_loss.png", dpi=300)
plt.show()

import torch
import numpy as np
from rl.models.tasq_bc.bc_actor import BCActor
import torch.nn as nn

# === Load your model ===
MODEL_PATH = "rl/models/tasq_bc/goal_directed/tasq_bc_ep200_20250706-073320.pt" 
DEVICE = "cpu"
OBS_DIM = 37 * 2 + 5 * 27  # Update if needed

model = BCActor(obs_dim=OBS_DIM)
model.initialize_weights_from_checkpoint(MODEL_PATH, device=DEVICE)
model.eval()

# === Load dataset ===
demo_states = np.load("rl/datasets/goal_directed/demo_states_0.npy")
demo_actions = np.load("rl/datasets/goal_directed/demo_actions_0.npy")

# === Construct example input ===
partial_idx = np.r_[
    0:3,    # base pos
    3:7,    # base orn
    13:25,  # pressures
    25:29,  # suctions
    33:37   # goal_pos and orn
]

full = demo_states[400]  # shape: (37,)
fullm1 = demo_states[400 - 1] # shape: (37,)
partial1 = demo_states[400 - 40][partial_idx]  # shape: (X,)
partial2 = demo_states[400 - 80][partial_idx]  # shape: (X,)
partial3= demo_states[400 - 120][partial_idx]  # shape: (X,)
partial4 = demo_states[400 - 160][partial_idx]  # shape: (X,)
partial5 = demo_states[400 - 200][partial_idx]  # shape: (X,)

# Build final input state
state = np.concatenate([
    full, fullm1,
    partial1, partial2, partial3, partial4, partial5
])  # shape: (209,)

print("State shape:", state.shape)

# === Run the model ===
with torch.no_grad():
    input_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 209)
    pressures, suctions = model(input_tensor)
    pressures = pressures.squeeze(0).detach().numpy()
    suctions = suctions.squeeze(0).detach().numpy()

# === Retrieve ground truth action ===
action = torch.tensor(demo_actions[400], dtype=torch.float32)

# Convert model outputs to torch tensors
predicted = torch.tensor(np.concatenate([pressures, suctions]), dtype=torch.float32)

# Compute MSE loss
mse_loss = nn.MSELoss()
MSE_Loss = mse_loss(predicted, action)

# === Print outputs ===
print("Input state:", state)
print("Predicted pressures:", pressures)
print("Predicted suctions:", suctions)
print("Ground truth action:", action)
print("MSE_Loss:", MSE_Loss)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- Configurations ---
MODEL_DIR = "rl/models/tasq_bc"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "tasq_bc"

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load demonstration data ---
# Assume you have numpy arrays saved as demo_states.npy and demo_actions.npy
states = np.load("demo_states.npy")   # shape: (N, obs_dim)
actions = np.load("demo_actions.npy") # shape: (N, action_dim)

# Convert to torch tensors
states_t = torch.from_numpy(states).float()
actions_t = torch.from_numpy(actions).float()

dataset = TensorDataset(states_t, actions_t)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Define the network matching SAC actor ---
class BCActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Because SAC actor output is in [-1,1] for pressure delta + [0,1] for suction
        )
        
    def forward(self, x):
        return self.net(x)

# --- Instantiate model ---
obs_dim = states.shape[1]
action_dim = actions.shape[1]
model = BCActor(obs_dim, action_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- Training loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch_states, batch_actions in dataloader:
        batch_states = batch_states.to(DEVICE)
        batch_actions = batch_actions.to(DEVICE)

        optimizer.zero_grad()
        pred_actions = model(batch_states)
        loss = criterion(pred_actions, batch_actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_states.size(0)
    
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.6f}")

# --- Save model ---
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{timestamp}.pt")
torch.save(model.state_dict(), model_path)
print(f"Behavior cloning model saved at {model_path}")

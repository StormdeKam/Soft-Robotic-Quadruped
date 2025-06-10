import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# --- Configurations ---
MODEL_DIR = "rl/models/tasq_bc"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "tasq_bc"

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.2

print("Started behavior cloning training using ", DEVICE)

# --- Load demonstration data ---
states = np.load("rl/datasets/demo_states.npy")   # shape: (N, obs_dim)
actions = np.load("rl/datasets/demo_actions.npy") # shape: (N, action_dim)

# Stack frames
N = 4  # Number of frames to stack
stacked_states = []
stacked_actions = []

for i in range(N - 1, len(states)):
    stacked_obs = states[i - N + 1:i + 1].reshape(-1)  # shape: (obs_dim * N,)
    stacked_states.append(stacked_obs)
    stacked_actions.append(actions[i])  # Use action at time i (last obs)

stacked_states = np.array(stacked_states)
stacked_actions = np.array(stacked_actions)

# Convert to torch tensors
states_t = torch.from_numpy(stacked_states).float()
actions_t = torch.from_numpy(stacked_actions).float()

# Create full dataset
full_dataset = TensorDataset(states_t, actions_t)

# --- Split dataset ---
total_size = len(full_dataset)
val_size = int(total_size * VAL_SPLIT)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

# --- Instantiate model ---
obs_dim = stacked_states.shape[1]  # = original_obs_dim * N
action_dim = stacked_actions.shape[1]
model = BCActor(obs_dim, action_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- Training loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0
    for batch_states, batch_actions in train_loader:
        batch_states = batch_states.to(DEVICE)
        batch_actions = batch_actions.to(DEVICE)

        optimizer.zero_grad()
        pred_actions = model(batch_states)
        loss = criterion(pred_actions, batch_actions)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_states.size(0)
    
    avg_train_loss = total_train_loss / train_size

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_states, val_actions in val_loader:
            val_states = val_states.to(DEVICE)
            val_actions = val_actions.to(DEVICE)

            val_pred = model(val_states)
            val_loss = criterion(val_pred, val_actions)
            total_val_loss += val_loss.item() * val_states.size(0)
    
    avg_val_loss = total_val_loss / val_size
    print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

# --- Save model ---
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_{timestamp}.pt")
torch.save(model.state_dict(), model_path)
print(f"Behavior cloning model saved at {model_path}")

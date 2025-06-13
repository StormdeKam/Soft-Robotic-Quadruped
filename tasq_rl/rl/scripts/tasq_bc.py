import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from rl.utils.dataset_stacker import StackedDataset

# --- Configuration ---
MODEL_DIR = "rl/models/tasq_bc"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "tasq_bc"
LOG_FILE = os.path.join(MODEL_DIR, f"{MODEL_NAME}_log.txt")

BATCH_SIZE = 64
LEARNING_RATE = 1e-4 #original 3e-4
EPOCHS = 400
SAVE_INTERVAL = 50
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Optional: Cap GPU memory usage (e.g., 90%)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)

print("Started behavior cloning training using", DEVICE)

# --- Load demonstration data efficiently ---
print("Loading dataset...")
dataset = StackedDataset(
    state_path="rl/datasets/demo_states3.npy",
    action_path="rl/datasets/demo_actions3.npy",
    N=4
)

# --- Datasets & loader ---
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

# --- Define model ---
class BCActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self.net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.Tanh()
        )


        #self.initialize_weights()

    def initialize_weights(self):
        weight = self.fc1.weight.data
        bias = self.fc1.bias.data

        # Feature index ranges
        base_pos_idx = slice(0, 3)
        base_quat_idx = slice(3, 7)
        base_vel_idx = slice(7, 13)
        pressure_idx = slice(13, 25)
        suction_idx = slice(25, 29)
        grf_idx = slice(29, 41)
        time_idx = 42

        # Smaller weights for base and GRF features
        weight[:, base_pos_idx] = 0.01 * torch.randn_like(weight[:, base_pos_idx])
        weight[:, base_quat_idx] = 0.01 * torch.randn_like(weight[:, base_quat_idx])
        weight[:, base_vel_idx] = 0.01 * torch.randn_like(weight[:, base_vel_idx])
        weight[:, grf_idx] = 0.01 * torch.randn_like(weight[:, grf_idx])

        # Larger weights for pressures, suctions and time
        weight[:, pressure_idx] = 0.1 * torch.randn_like(weight[:, pressure_idx])
        weight[:, suction_idx] = 0.1 * torch.randn_like(weight[:, suction_idx])
        weight[:, time_idx] = 0.1 * torch.randn_like(weight[:, grf_idx])


        # Optional: zero out the bias
        bias.zero_()


    def forward(self, x):
        return self.net(x)

obs_dim = dataset.obs_dim * dataset.N
action_dim = dataset.action_dim
model = BCActor(obs_dim, action_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- Log file setup ---
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,val_loss,model_path\n")

# --- Training loop ---
print("Beginning training...")
start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0

    # 🟦 tqdm here for training loop
    for batch_states, batch_actions in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training", leave=False):
        batch_states = batch_states.to(DEVICE, non_blocking=True)
        batch_actions = batch_actions.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        pred = model(batch_states)
        loss = criterion(pred, batch_actions)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_states.size(0)

    avg_train_loss = total_train_loss / train_size

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    # 🟦 tqdm here for validation loop
    with torch.no_grad():
        for val_states, val_actions in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation", leave=False):
            val_states = val_states.to(DEVICE, non_blocking=True)
            val_actions = val_actions.to(DEVICE, non_blocking=True)

            val_pred = model(val_states)
            val_loss = criterion(val_pred, val_actions)
            total_val_loss += val_loss.item() * val_states.size(0)

    avg_val_loss = total_val_loss / val_size

    # --- Logging ---
    elapsed = time.time() - start_time
    avg_epoch_time = elapsed / epoch
    remaining = avg_epoch_time * (EPOCHS - epoch)
    rem_hours = int(remaining // 3600)
    rem_mins = int((remaining % 3600) // 60)

    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | ETA: {rem_hours}h {rem_mins}m")

    # Save checkpoint
    if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_ep{epoch}_{timestamp}.pt")
        torch.save(model.state_dict(), model_path)
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.6f},{avg_val_loss:.6f},{model_path}\n")

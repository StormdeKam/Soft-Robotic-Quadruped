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
EPOCHS = 100
SAVE_INTERVAL = 25
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Optional: Cap GPU memory usage (e.g., 90%)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)

print("Started behavior cloning training using", DEVICE)

# --- Load demonstration data efficiently ---
print("Loading dataset...")
dataset = StackedDataset(
    state_path="rl/datasets/demo_states7.npy",
    action_path="rl/datasets/demo_actions7.npy",
    N=4
)

# --- Datasets & loader ---
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

# --- Define model ---
# --- Define RNN model ---
class RNNActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        last_out = out[:, -1, :]  # Use last time step's output
        return self.fc(last_out)

# --- Model and optimizer setup ---
obs_dim = dataset.obs_dim
seq_len = dataset.N
action_dim = dataset.action_dim

model = RNNActor(input_dim=obs_dim, hidden_dim=128, output_dim=action_dim).to(DEVICE)
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
        # For training and validation:
        batch_states = batch_states.view(batch_states.size(0), seq_len, obs_dim).to(DEVICE, non_blocking=True)
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
            # For training and validation:
            val_states = val_states.view(val_states.size(0), seq_len, obs_dim).to(DEVICE, non_blocking=True)
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

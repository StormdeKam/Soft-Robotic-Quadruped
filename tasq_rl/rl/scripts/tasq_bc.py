import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from rl.utils.dataset_stacker import StackedDataset
from rl.utils.interval_stacker import IntervalStackedDataset

# --- Configuration ---
MODEL_DIR = "rl/models/tasq_bc/goal_directed"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "tasq_bc"
LOG_FILE = os.path.join(MODEL_DIR, f"{MODEL_NAME}_log.txt")

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 200
SAVE_INTERVAL = 10
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)

print("Started behavior cloning training using", DEVICE)
"""
# Use exactly like before
dataset = StackedDataset(
    "rl/datasets/goal_directed/demo_states_merged.npy",
    "rl/datasets/goal_directed/demo_actions_merged.npy",
    N=4
)
"""
# Use exactly like before
dataset = IntervalStackedDataset(
    "rl/datasets/goal_directed/demo_states_merged.npy",
    "rl/datasets/goal_directed/demo_actions_merged.npy",
    "rl/datasets/goal_directed/demo_episode_lengths_merged.npy"
)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

# --- Model with two heads ---
class BCActor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.pressure_head = nn.Sequential(
            nn.Linear(256, 12),
            nn.Tanh()
        )
        self.suction_head = nn.Linear(256, 4)

    def forward(self, x):
        x = self.shared(x)
        pressures = self.pressure_head(x)       # Linear output for MSE
        suctions = self.suction_head(x)         # Logits for BCEWithLogitsLoss
        return pressures, suctions

    def initialize_weights_from_checkpoint(self, checkpoint_path, device="cpu"):
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        print(f"Loaded weights from {checkpoint_path}")
        return self

obs_dim = dataset.obs_dim * 2 + 5 * 27 # 2 previous full states and 5 partial states from history in intervals
model = BCActor(obs_dim).to(DEVICE)
"""
try:
    model = model.initialize_weights_from_checkpoint(os.path.join(MODEL_DIR, "tasq_bc_ep100_20250619-025729.pt"), DEVICE)
except Exception as e:
    print(f"Weight loading failed: {e}")
"""
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

# --- Logging ---
# Update log file header
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,train_loss_pressures,train_loss_suctions,val_loss,val_loss_pressures,val_loss_suctions,model_path\n")

# --- Training ---
print("Beginning training...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    total_train_loss_pressures = 0.0
    total_train_loss_suctions = 0.0

    for batch_states, batch_actions in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training", leave=False):
        batch_states = batch_states.to(DEVICE, non_blocking=True)
        batch_actions = batch_actions.to(DEVICE, non_blocking=True)

        target_pressures = batch_actions[:, :12]
        target_suctions = batch_actions[:, 12:16]

        optimizer.zero_grad()
        pred_pressures, pred_suctions = model(batch_states)
        loss_pressures = mse_loss(pred_pressures, target_pressures)
        loss_suctions = bce_loss(pred_suctions, target_suctions)
        loss = loss_pressures + loss_suctions
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_states.size(0)
        total_train_loss_pressures += loss_pressures.item() * batch_states.size(0)
        total_train_loss_suctions += loss_suctions.item() * batch_states.size(0)

    avg_train_loss = total_train_loss / train_size
    avg_train_loss_pressures = total_train_loss_pressures / train_size
    avg_train_loss_suctions = total_train_loss_suctions / train_size

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    total_val_loss_pressures = 0.0
    total_val_loss_suctions = 0.0

    with torch.no_grad():
        for val_states, val_actions in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation", leave=False):
            val_states = val_states.to(DEVICE, non_blocking=True)
            val_actions = val_actions.to(DEVICE, non_blocking=True)

            target_pressures = val_actions[:, :12]
            target_suctions = val_actions[:, 12:16]
            pred_pressures, pred_suctions = model(val_states)

            val_loss_pressures = mse_loss(pred_pressures, target_pressures)
            val_loss_suctions = bce_loss(pred_suctions, target_suctions)
            val_loss = val_loss_pressures + val_loss_suctions

            total_val_loss += val_loss.item() * val_states.size(0)
            total_val_loss_pressures += val_loss_pressures.item() * val_states.size(0)
            total_val_loss_suctions += val_loss_suctions.item() * val_states.size(0)

    avg_val_loss = total_val_loss / val_size
    avg_val_loss_pressures = total_val_loss_pressures / val_size
    avg_val_loss_suctions = total_val_loss_suctions / val_size

    scheduler.step(avg_val_loss)

    # --- ETA calculation ---
    elapsed = time.time() - start_time
    avg_epoch_time = elapsed / epoch
    remaining = avg_epoch_time * (EPOCHS - epoch)
    rem_hours = int(remaining // 3600)
    rem_mins = int((remaining % 3600) // 60)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"[Epoch {epoch}/{EPOCHS}] "
          f"Train: total={avg_train_loss:.6f}, pressures={avg_train_loss_pressures:.6f}, suctions={avg_train_loss_suctions:.6f} | "
          f"Val: total={avg_val_loss:.6f}, pressures={avg_val_loss_pressures:.6f}, suctions={avg_val_loss_suctions:.6f} | "
          f"LR: {current_lr:.2e} | ETA: {rem_hours}h {rem_mins}m")

    # --- Always log to file (even if no checkpoint is saved) ---
    model_path = ""
    if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_ep{epoch}_{timestamp}.pt")
        torch.save(model.state_dict(), model_path)

    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_train_loss:.6f},{avg_train_loss_pressures:.6f},{avg_train_loss_suctions:.6f},"
                f"{avg_val_loss:.6f},{avg_val_loss_pressures:.6f},{avg_val_loss_suctions:.6f},{model_path}\n")


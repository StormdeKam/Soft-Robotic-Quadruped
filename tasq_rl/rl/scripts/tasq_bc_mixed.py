import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from tqdm import tqdm
from rl.utils.framestacking.dataset_stacker import StackedDataset
from rl.models.architecture.actor import BCActor

# --- Configuration ---
MODEL_DIR = "rl/models/tasq_bc/minimal_viable_V6/mixed_dataset"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "tasq_bc"
LOG_FILE = os.path.join(MODEL_DIR, f"{MODEL_NAME}_log.txt")

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 150
SAVE_INTERVAL = 10
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)

print("Started behavior cloning training using", DEVICE)

OBSERVED_INDICES = np.r_[
    3:7,    # base orientation
    10:13,  # base_angular_velocity
    13:25,  # pressures
    25:29,  # suctions
    29:33,  # grfs
    33:37   # goal_pos and orientation
]

# --- Load datasets ---
real_dataset = StackedDataset(
    "rl/datasets/minimal_viable_v6/real_datasets/0.5xspeed/real_states_merged.npy",
    "rl/datasets/minimal_viable_v6/real_datasets/0.5xspeed/real_actions_merged.npy",
    N=4,
    observed_indices=OBSERVED_INDICES
)

sim_dataset = StackedDataset(
    "rl/datasets/minimal_viable_v6/demo_states_merged.npy",
    "rl/datasets/minimal_viable_v6/demo_actions_merged.npy",
    N=4,
    observed_indices=OBSERVED_INDICES
)

# --- Merge datasets ---
full_dataset = ConcatDataset([sim_dataset, real_dataset])
dataset_lengths = [len(sim_dataset), len(real_dataset)]

# --- Weighted sampler for 50% real contribution ---
weights = np.concatenate([
    np.ones(dataset_lengths[0]),  # sim dataset
    np.ones(dataset_lengths[1]) * (dataset_lengths[0] / dataset_lengths[1])  # boost real
])
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# --- Split into train/val ---
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

# Use indices to split
all_indices = np.arange(len(full_dataset))
np.random.shuffle(all_indices)
train_indices = all_indices[:train_size]
val_indices = all_indices[train_size:]

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

# --- DataLoaders ---
train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True, num_workers=4)
val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=4)

# --- Model ---
obs_dim = len(OBSERVED_INDICES) * 4  # N=4
model = BCActor(obs_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

# --- Logging ---
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

        # replace any NaNs in input with 0
        batch_states = torch.nan_to_num(batch_states, nan=0.0)
        batch_actions = torch.nan_to_num(batch_actions, nan=0.0)

        target_pressures = batch_actions[:, :12]
        target_suctions = batch_actions[:, 12:16]

        optimizer.zero_grad()
        pred_pressures, pred_suctions = model(batch_states)
        loss_pressures = mse_loss(pred_pressures, target_pressures)
        loss_suctions = mse_loss(pred_suctions, target_suctions)
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

            # replace NaNs
            val_states = torch.nan_to_num(val_states, nan=0.0)
            val_actions = torch.nan_to_num(val_actions, nan=0.0)

            target_pressures = val_actions[:, :12]
            target_suctions = val_actions[:, 12:16]
            pred_pressures, pred_suctions = model(val_states)

            val_loss_pressures = mse_loss(pred_pressures, target_pressures)
            val_loss_suctions = mse_loss(pred_suctions, target_suctions)
            val_loss = val_loss_pressures + val_loss_suctions

            total_val_loss += val_loss.item() * val_states.size(0)
            total_val_loss_pressures += val_loss_pressures.item() * val_states.size(0)
            total_val_loss_suctions += val_loss_suctions.item() * val_states.size(0)

    avg_val_loss = total_val_loss / val_size
    avg_val_loss_pressures = total_val_loss_pressures / val_size
    avg_val_loss_suctions = total_val_loss_suctions / val_size

    scheduler.step(avg_val_loss)

    # --- ETA ---
    elapsed = time.time() - start_time
    avg_epoch_time = elapsed / epoch
    remaining = avg_epoch_time * (EPOCHS - epoch)
    rem_hours = int(remaining // 3600)
    rem_mins = int((remaining % 3600) // 60)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"[Epoch {epoch}/{EPOCHS}] "
          f"Train: total={avg_train_loss:.12f}, pressures={avg_train_loss_pressures:.12f}, suctions={avg_train_loss_suctions:.12f} | "
          f"Val: total={avg_val_loss:.12f}, pressures={avg_val_loss_pressures:.12f}, suctions={avg_val_loss_suctions:.12f} | "
          f"LR: {current_lr:.2e} | ETA: {rem_hours}h {rem_mins}m")

    # --- Save checkpoint ---
    model_path = ""
    if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_ep{epoch}.pt")
        torch.save(model.state_dict(), model_path)

    # --- Log ---
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_train_loss:.12f},{avg_train_loss_pressures:.12f},{avg_train_loss_suctions:.12f},"
                f"{avg_val_loss:.12f},{avg_val_loss_pressures:.12f},{avg_val_loss_suctions:.12f},{model_path}\n")

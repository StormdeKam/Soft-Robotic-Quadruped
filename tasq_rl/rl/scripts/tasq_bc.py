import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from rl.utils.dataset_stacker import StackedDataset

# --- Configuration ---
MODEL_DIR = "rl/models/tasq_bc/RNN"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = "tasq_bc"
LOG_FILE = os.path.join(MODEL_DIR, f"{MODEL_NAME}_log.txt")

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100
SAVE_INTERVAL = 5
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)

print("Started behavior cloning training using", DEVICE)

# --- Load demonstration data efficiently ---
print("Loading dataset...")
dataset = StackedDataset(
    state_path="rl/datasets/demo_states8.npy",
    action_path="rl/datasets/demo_actions8.npy",
    N=10
)

# --- Datasets & loader ---
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

# --- Define RNN model with two heads ---
class RNNActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, pressure_dim=12, suction_dim=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )
        self.pressure_head = nn.Linear(256, pressure_dim)  # MSE output
        self.suction_head = nn.Linear(256, suction_dim)    # BCEWithLogits output

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        shared = self.shared_fc(last_out)
        pressures = self.pressure_head(shared)
        suctions = self.suction_head(shared)
        return pressures, suctions

# --- Model and optimizer setup ---
obs_dim = dataset.obs_dim
seq_len = dataset.N
action_dim = dataset.action_dim

model = RNNActor(input_dim=obs_dim, hidden_dim=128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,        # Reduce LR by this factor
    patience=5,        # Wait 5 epochs before reducing
    min_lr=1e-6       # Don't go lower than this
)

# --- Log file setup ---
with open(LOG_FILE, "w") as f:
    f.write("epoch,train_loss,val_loss,model_path\n")

# --- Training loop ---
print("Beginning training...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0

    for batch_states, batch_actions in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training", leave=False):
        batch_states = batch_states.view(batch_states.size(0), seq_len, obs_dim).to(DEVICE, non_blocking=True)
        batch_actions = batch_actions.to(DEVICE, non_blocking=True)

        target_pressures = batch_actions[:, :12]
        target_suctions = batch_actions[:, 12:]

        optimizer.zero_grad()
        pred_pressures, pred_suctions = model(batch_states)

        loss_pressures = mse_loss(pred_pressures, target_pressures)
        loss_suctions = bce_loss(pred_suctions, target_suctions)
        loss = loss_pressures + loss_suctions

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_states.size(0)

    avg_train_loss = total_train_loss / train_size

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for val_states, val_actions in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation", leave=False):
            val_states = val_states.view(val_states.size(0), seq_len, obs_dim).to(DEVICE, non_blocking=True)
            val_actions = val_actions.to(DEVICE, non_blocking=True)

            target_pressures = val_actions[:, :12]
            target_suctions = val_actions[:, 12:]

            pred_pressures, pred_suctions = model(val_states)

            val_loss_pressures = mse_loss(pred_pressures, target_pressures)
            val_loss_suctions = bce_loss(pred_suctions, target_suctions)
            val_loss = val_loss_pressures + val_loss_suctions

            total_val_loss += val_loss.item() * val_states.size(0)

    avg_val_loss = total_val_loss / val_size
    scheduler.step(avg_val_loss)
    # --- Logging ---
    elapsed = time.time() - start_time
    avg_epoch_time = elapsed / epoch
    remaining = avg_epoch_time * (EPOCHS - epoch)
    rem_hours = int(remaining // 3600)
    rem_mins = int((remaining % 3600) // 60)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e} | ETA: {rem_hours}h {rem_mins}m") 

    # Save checkpoint
    if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_ep{epoch}_{timestamp}.pt")
        torch.save(model.state_dict(), model_path)
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.6f},{avg_val_loss:.6f},{model_path}\n")

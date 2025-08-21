import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# --- Define the model ---
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

# --- Load model ---
model = BCActor(obs_dim=41*4)
model.load_state_dict(torch.load("rl/models/tasq_bc/2_heads/tasq_bc_ep100_20250620-002326.pt"))
model.eval()

# --- Load states and sample 25 stacked observations ---
states = np.load("rl/datasets/random_gait_cycle_start/demo_states_19.npy")
stack_size = 4
sample_size = 25
state_dim = 41

np.random.seed(42)
max_start = len(states) - stack_size
indices = np.random.choice(max_start, sample_size, replace=False)

stacked_inputs = np.array([states[i:i+stack_size].flatten() for i in indices])  # shape: (25, 168)

# --- Convert to torch and enable gradients ---
inputs = torch.tensor(stacked_inputs, dtype=torch.float32, requires_grad=True)

# --- Forward pass ---
pressures, suctions = model(inputs)
outputs = torch.cat([pressures, suctions], dim=1)  # shape: (25, 16)

# --- Compute saliency (mean gradient of each output w.r.t each input) ---
saliency = torch.zeros(outputs.shape[1], inputs.shape[1])  # shape: (16, 168)

for i in range(outputs.shape[1]):  # loop over output dims
    model.zero_grad()
    grad_outputs = torch.zeros_like(outputs)
    grad_outputs[:, i] = 1.0
    outputs.backward(gradient=grad_outputs, retain_graph=True)
    saliency[i] = inputs.grad.abs().mean(dim=0)
    inputs.grad.zero_()

saliency = saliency.numpy()  # shape: (16, 168)

# --- Create input labels ---
feature_groups = ["pos", "ori", "vel", "pres", "suct", "grf"]
group_sizes = [3, 4, 6, 12, 4, 12]

input_labels = []
for frame in range(4):
    for group, size in zip(feature_groups, group_sizes):
        input_labels += [f"{group}{i+1}_f{frame+1}" for i in range(size)]

output_labels = ["pressure 1", 
                "pressure 2", 
                "pressure 3", 
                "pressure 4", 
                "pressure 5",
                "pressure 6", 
                "pressure 7",
                "pressure 8",
                "pressure 9",
                "pressure 10",
                "pressure 11",
                "pressure 12",
                "suction 1",
                "suction 2",
                "suction 3",
                "suction 4"]

# --- Plot ---
plt.figure(figsize=(18, 6))
sns.heatmap(saliency, cmap="coolwarm", xticklabels=input_labels, yticklabels=output_labels)
plt.title("Saliency Map: Mean ∂Output / ∂Input over 25 Samples")
plt.xlabel("Input Features (framestacked)")
plt.ylabel("Output Neurons (pressures + suctions)")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.show()

# --- Aggregate saliency across frames ---
saliency = saliency.reshape(16, 4, 41)  # (outputs, frames, features)
saliency = saliency.mean(axis=1)  # shape: (16, 41)

# --- Update input labels to one per feature ---
input_labels = []
for group, size in zip(feature_groups, group_sizes):
    input_labels += [f"{group}{i+1}" for i in range(size)]

# --- Plot ---
plt.figure(figsize=(14, 6))
sns.heatmap(saliency, cmap="coolwarm", xticklabels=input_labels, yticklabels=output_labels)
plt.title("Saliency Map: Mean ∂Output / ∂Input over Frames (Averaged Across 4 Frames)")
plt.xlabel("Input Features (frame-averaged)")
plt.ylabel("Output Neurons (pressures + suctions)")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.show()



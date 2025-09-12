import torch
import numpy as np
from tasq_rl.rl.models.architecture.actor import BCActor
from rl.scripts.tasq_sac import SACActorWithBC
from rl.scripts.tasq_env import TASQEnv

# -----------------------------
# Config
# -----------------------------
bc_model_path = "rl/models/tasq_bc/minimal_viable_V2/tasq_bc_ep150_20250901-133944.pt"
sac_actor_path = "rl/models/tasq_sac/cloned_from_bc/tasq_sac_actor.pt"  # SAC actor saved as .pt
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Create TASQEnv and get obs shape
# -----------------------------
env = TASQEnv(
    urdf_path="rl/urdfs/TASQ_n7.urdf",
    no_links=7,
    gui=False,
    frame_stack=4
)
obs, _ = env.reset()
obs_dim = obs.shape[0]   # already includes frame stacking
act_dim = 16             # 12 pressures + 4 suctions

# -----------------------------
# Load BC model
# -----------------------------
bc_model = BCActor(obs_dim).to(device)
bc_model.load_state_dict(torch.load(bc_model_path, map_location=device))
bc_model.eval()

# -----------------------------
# Load SAC actor (BC-initialized clone)
# -----------------------------
sac_actor = SACActorWithBC(obs_dim=obs_dim, act_dim=act_dim, device=device)
sac_actor.load_state_dict(torch.load(sac_actor_path, map_location=device))
sac_actor.eval()

# -----------------------------
# Compare outputs
# -----------------------------
obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)

with torch.no_grad():
    # BC output
    bc_pressures, bc_suctions = bc_model(obs_tensor)
    bc_action = torch.cat([bc_pressures, bc_suctions], dim=-1).squeeze(0)

    # SAC actor output
    sac_action = sac_actor(obs_tensor).squeeze(0)

# Compute max absolute difference
diff = torch.abs(bc_action - sac_action).max().item()
print(f"Max absolute difference between BC and SAC actor: {diff:.24f}")

# Optional: elementwise inspection
print("BC actions: ", bc_action.cpu().numpy())
print("SAC actions:", sac_action.cpu().numpy())

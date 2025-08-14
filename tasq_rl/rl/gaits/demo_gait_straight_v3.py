import numpy as np
import pandas as pd

# --- Load CSV ---
csv_file = "rl/gaits/GAIT2.csv"
data = pd.read_csv(csv_file, sep=';', header=None, encoding='utf-8-sig', dtype=float, decimal=',')
pressures = data.to_numpy()

# First 12 columns: pressures
pressures = data.iloc[:, :12].to_numpy()  # shape (num_steps, 12)
# Last 4 columns: suction (100 -> True, 0 -> False)
suction_raw = data.iloc[:, 12:16].to_numpy()
suction = suction_raw == 100

# Compute pressure gradients (delta between steps)
pressure_gradients = np.diff(pressures, axis=0, prepend=pressures[0:1, :])
# Reshape gradients to (num_steps, 4, 3)
pressure_gradients = pressure_gradients.reshape(-1, 4, 3)

num_steps = pressure_gradients.shape[0]

# --- Callback functions ---

def gait_pressures(step, mode=None):
    """Returns pressure gradients for a given step"""
    step_mod = step % num_steps
    return pressure_gradients[step_mod]

def gait_suction(step, mode=None):
    """Returns suction states for a given step"""
    step_mod = step % num_steps
    return suction[step_mod]

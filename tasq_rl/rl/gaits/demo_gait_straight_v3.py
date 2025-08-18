import numpy as np
import pandas as pd

# === Configuration ===
real_world_freq = 40.0   # Hz (your real robot playback: 25 ms -> 40 Hz)
sim_control_freq = 12.0  # Hz (your simulation control loop)

# --- Load CSV ---
csv_file = "rl/gaits/GAIT2.csv"
data = pd.read_csv(
    csv_file,
    sep=';',
    header=None,
    encoding='utf-8-sig',
    decimal=','
)

# First 12 columns: absolute pressures (from CSV)
pressures = data.iloc[:, :12].to_numpy(dtype=float)   # shape (N, 12)

# Last 4 columns: suction (100 -> True, 0 -> False)
suction_raw = data.iloc[:, 12:16].to_numpy(dtype=float)
suction = suction_raw == 100                           # shape (N, 4)

N = pressures.shape[0]
assert N >= 2, "CSV must contain at least 2 rows of gait data."

# --- Cyclic per-interval gradients (between i -> i+1, wrapping) ---
# d[i] = pressures[(i+1) % N] - pressures[i]
d = pressures[(np.arange(N) + 1) % N] - pressures

# --- Mapping: how many real-world gait steps per sim step ---
step_factor = real_world_freq / sim_control_freq  # e.g., 40/24 ≈ 1.666...

def _delta_over_interval(x0: float, x1: float) -> np.ndarray:
    """
    Integrate per-interval gradients d over the fractional path from x0 to x1
    on the cyclic index domain [0, N), returning a (12,) delta.
    - x0, x1 are fractional indices in [0, N), can wrap around.
    """
    total = np.zeros((12,), dtype=float)

    # Unwrap so that x1 >= x0 in the same linearized domain
    if x1 < x0:
        x1 += N

    i0 = int(np.floor(x0)) % N
    f0 = x0 - np.floor(x0)        # fractional offset within interval i0 -> i0+1

    i1 = int(np.floor(x1)) % N
    f1 = x1 - np.floor(x1)

    # If entirely within a single interval (no crossing to the next integer)
    if int(np.floor(x1)) == int(np.floor(x0)) and x1 < x0 + 1.0:
        frac = x1 - x0
        total += frac * d[i0]
        return total

    # Part 1: remainder of the starting interval
    total += (1.0 - f0) * d[i0]

    # Part 2: any full intervals between start and end
    full_count = int(np.floor(x1)) - int(np.floor(x0)) - 1
    if full_count < 0:
        full_count += N

    # Accumulate full gradients
    for k in range(full_count):
        idx = (i0 + 1 + k) % N
        total += d[idx]

    # Part 3: initial fraction of the ending interval
    total += f1 * d[i1]

    return total

def _suction_at_fractional_step(x: float) -> np.ndarray:
    """
    Choose a suction state at fractional index x by midpoint switching.
    This keeps it discrete while aligning with the fractional timeline.
    """
    base_idx = int(np.floor(x)) % N
    next_idx = (base_idx + 1) % N
    frac = x - np.floor(x)

    if np.array_equal(suction[base_idx], suction[next_idx]):
        return suction[base_idx]
    else:
        # Switch halfway through the interval
        return suction[base_idx] if frac < 0.5 else suction[next_idx]

# --- Public callbacks expected by your simulator ---

def gait_pressures(step, mode=None):
    """
    Return the *delta pressures* (gradients) to apply this sim step
    using fractional integration of real-world step gradients so that
    totals match the real gait even when frequencies differ.
    Output shape: (4, 3)
    """
    # From previous sim step's fractional position to current
    if step <= 0:
        x0 = 0.0
        x1 = (step_factor) % N
    else:
        x0 = ((step - 1) * step_factor) % N
        x1 = (step * step_factor) % N

    delta_12 = _delta_over_interval(x0, x1)  # (12,)
    return delta_12.reshape(4, 3)

def gait_suction(step, mode=None):
    """
    Return suction states aligned to the current fractional position.
    """
    x = (step * step_factor) % N
    return _suction_at_fractional_step(x)

import pandas as pd
import numpy as np
import os
import re

def process_and_merge():
    base_path = "rl/datasets/minimal_viable_v6/real_datasets/0.5xspeed"

    state_arrays = []
    action_arrays = []

    # Collect all state and action files
    all_files = os.listdir(base_path)
    state_files = sorted([f for f in all_files if re.match(r"real_states_\d+\.csv", f)],
                         key=lambda x: int(re.findall(r"\d+", x)[0]))
    action_files = sorted([f for f in all_files if re.match(r"real_actions_\d+\.csv", f)],
                          key=lambda x: int(re.findall(r"\d+", x)[0]))

    # --- Process states ---
    for f in state_files:
        input_path = os.path.join(base_path, f)
        df_states = pd.read_csv(input_path, header=None)

        if df_states.shape[1] != 31:
            print(f"Warning: {f} has {df_states.shape[1]} columns (expected 31).")

        # Add NaN columns
        nan_cols_before = pd.DataFrame(np.nan, index=df_states.index, columns=[f"NaN_before_{i}" for i in range(3)])
        nan_cols_middle = pd.DataFrame(np.nan, index=df_states.index, columns=[f"NaN_middle_{i}" for i in range(3)])

        df_states_new = pd.concat(
            [nan_cols_before, df_states.iloc[:, :7], nan_cols_middle, df_states.iloc[:, 7:]],
            axis=1
        )

        state_arrays.append(df_states_new.to_numpy())
        print(f"Processed {f}")

    # --- Process actions ---
    for f in action_files:
        input_path = os.path.join(base_path, f)
        df_actions = pd.read_csv(input_path, header=None)
        action_arrays.append(df_actions.to_numpy())
        print(f"Processed {f}")

    # --- Merge and save ---
    states_merged = np.vstack(state_arrays)
    actions_merged = np.vstack(action_arrays)

    np.save(os.path.join(base_path, "real_states_merged.npy"), states_merged)
    np.save(os.path.join(base_path, "real_actions_merged.npy"), actions_merged)

    print(f"Saved merged states to real_states_merged.npy with shape {states_merged.shape}")
    print(f"Saved merged actions to real_actions_merged.npy with shape {actions_merged.shape}")


if __name__ == "__main__":
    process_and_merge()

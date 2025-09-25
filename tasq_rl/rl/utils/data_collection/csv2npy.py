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

    if len(state_files) != len(action_files):
        print(f"Warning: different number of state ({len(state_files)}) and action ({len(action_files)}) files")

    # --- Process each pair ---
    for s_file, a_file in zip(state_files, action_files):
        # --- STATES ---
        df_states = pd.read_csv(os.path.join(base_path, s_file), header=None)

        # Replace any NaN in raw data with 0
        if df_states.isna().any().any():
            print(f"Warning: {s_file} contains NaNs. Replacing with 0.")
            df_states = df_states.fillna(0.0)

        if df_states.shape[1] != 31:
            print(f"Warning: {s_file} has {df_states.shape[1]} columns (expected 31).")

        # Add NaN columns at 0:3 and 7:10
        nan_before = pd.DataFrame(np.nan, index=df_states.index, columns=[f"NaN_before_{i}" for i in range(3)])
        nan_middle = pd.DataFrame(np.nan, index=df_states.index, columns=[f"NaN_middle_{i}" for i in range(3)])

        # Construct new dataframe: first 3 NaNs, original 0-6, next 3 NaNs, remaining columns
        df_states_new = pd.concat(
            [nan_before, df_states.iloc[:, :4], nan_middle, df_states.iloc[:, 4:]],
            axis=1
        )

        # --- ACTIONS ---
        df_actions = pd.read_csv(os.path.join(base_path, a_file), header=None)

        # Replace NaN in actions
        if df_actions.isna().any().any():
            print(f"Warning: {a_file} contains NaNs. Replacing with 0.")
            df_actions = df_actions.fillna(0.0)

        # Trim to minimum length
        min_len = min(len(df_states_new), len(df_actions))
        if len(df_states_new) != len(df_actions):
            print(f"Warning: {s_file} and {a_file} have different row counts ({len(df_states_new)} vs {len(df_actions)}). Trimming to {min_len} rows.")
        df_states_new = df_states_new.iloc[:min_len]
        df_actions = df_actions.iloc[:min_len]

        state_arrays.append(df_states_new.to_numpy())
        action_arrays.append(df_actions.to_numpy())

    # --- Merge all ---
    states_merged = np.vstack(state_arrays)
    actions_merged = np.vstack(action_arrays)

    np.save(os.path.join(base_path, "real_states_merged.npy"), states_merged)
    np.save(os.path.join(base_path, "real_actions_merged.npy"), actions_merged)

    print(f"Saved merged states to real_states_merged.npy with shape {states_merged.shape}")
    print(f"Saved merged actions to real_actions_merged.npy with shape {actions_merged.shape}")


if __name__ == "__main__":
    process_and_merge()

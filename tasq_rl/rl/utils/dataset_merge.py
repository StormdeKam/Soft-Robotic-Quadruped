import numpy as np
import os
import re

def load_sorted_files(folder, prefix):
    # List all files with prefix and extract chunk numbers
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.npy')]
    # Extract the number suffix (e.g. demo_states_3.npy -> 3)
    def extract_num(f):
        match = re.search(rf"{prefix}_(\d+)\.npy", f)
        return int(match.group(1)) if match else -1
    files_sorted = sorted(files, key=extract_num)
    return [os.path.join(folder, f) for f in files_sorted]

def merge_npy(folder, prefix, output_name):
    files = load_sorted_files(folder, prefix)
    if not files:
        print(f"No files found with prefix {prefix} in {folder}")
        return
    print(f"Found {len(files)} files for {prefix}. Loading and concatenating...")

    arrays = [np.load(f) for f in files]
    merged_array = np.concatenate(arrays, axis=0)
    out_path = os.path.join(folder, output_name)
    np.save(out_path, merged_array)
    print(f"Saved merged {prefix} to {out_path} with shape {merged_array.shape}")

if __name__ == "__main__":
    dataset_folder = "rl/datasets/no_steady_states"  # Adjust to your folder
    
    merge_npy(dataset_folder, "demo_states", "demo_states_merged.npy")
    merge_npy(dataset_folder, "demo_actions", "demo_actions_merged.npy")

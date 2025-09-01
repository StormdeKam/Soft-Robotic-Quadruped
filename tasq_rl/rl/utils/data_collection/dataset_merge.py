import numpy as np
import os
import re

def load_sorted_files(folder, prefix, min_index=0, max_index=float('inf')):
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.npy')]
    def extract_num(f):
        match = re.search(rf"{prefix}_(\d+)\.npy", f)
        return int(match.group(1)) if match else -1
    # Filter files within range
    filtered_files = [f for f in files 
                      if min_index <= extract_num(f) <= max_index and extract_num(f) >= 0]
    return sorted(filtered_files, key=extract_num)

def merge_npy_safely(folder, prefix, output_name, min_index=0, max_index=float('inf')):
    files = load_sorted_files(folder, prefix, min_index, max_index)
    if not files:
        print(f"No files found for {prefix} in range [{min_index}, {max_index}]")
        return

    # First pass: compute total shape
    total_rows = 0
    shape_last = None
    for f in files:
        arr = np.load(os.path.join(folder, f), mmap_mode='r')
        total_rows += arr.shape[0]
        shape_last = arr.shape[1:]
    full_shape = (total_rows, *shape_last)
    print(f"Total shape for {prefix} [{min_index}-{max_index}]: {full_shape}")

    # Allocate final big array
    merged_array = np.empty(full_shape, dtype=arr.dtype)

    # Second pass: fill in
    current_row = 0
    for f in files:
        arr = np.load(os.path.join(folder, f))
        merged_array[current_row : current_row + arr.shape[0]] = arr
        current_row += arr.shape[0]
        print(f"Added {f}, new cursor: {current_row}/{total_rows}")

    # Save to .npy
    out_path = os.path.join(folder, output_name)
    np.save(out_path, merged_array)
    print(f"✅ Saved merged array to {out_path} with shape {merged_array.shape}")

# Usage
if __name__ == "__main__":
    dataset_folder = "rl/datasets/minimal_viable"

    # Merge states
    merge_npy_safely(dataset_folder, "demo_states", "demo_states_merged.npy", min_index=0, max_index=20)
    #merge_npy_safely(dataset_folder, "demo_states", "demo_states_21_40.npy", min_index=21, max_index=40)
    #merge_npy_safely(dataset_folder, "demo_states", "demo_states_41_60.npy", min_index=41, max_index=60)
    #merge_npy_safely(dataset_folder, "demo_states", "demo_states_61_79.npy", min_index=61, max_index=79)

    # Merge actions
    merge_npy_safely(dataset_folder, "demo_actions", "demo_actions_merged.npy", min_index=0, max_index=20)
    #merge_npy_safely(dataset_folder, "demo_actions", "demo_actions_21_40.npy", min_index=21, max_index=40)
    #merge_npy_safely(dataset_folder, "demo_actions", "demo_actions_41_60.npy", min_index=41, max_index=60)
    #merge_npy_safely(dataset_folder, "demo_actions", "demo_actions_61_79.npy", min_index=61, max_index=79)

    # Merge episode lengths
    merge_npy_safely(dataset_folder, "demo_episode_lengths", "demo_episode_lengths_merged.npy", min_index=0, max_index=20)
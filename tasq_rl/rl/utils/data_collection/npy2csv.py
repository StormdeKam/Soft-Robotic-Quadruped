import numpy as np
import csv
import os

def npy_to_csv(npy_file, csv_file=None):
    # Load the NumPy array
    array = np.load(npy_file, allow_pickle=False)
    # Print shape and dtype
    print(f"Loaded: {npy_file}")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"First item (preview): {array[0] if len(array) > 0 else 'Empty'}")
    #"""
    # Set default CSV file name if not provided
    if csv_file is None:
        csv_file = os.path.splitext(npy_file)[0] + ".csv"

    # Check if it's 1D or 2D for CSV compatibility
    if array.ndim == 1:
        array = array.reshape(-1, 1)  # convert to 2D column
    
    # Save as CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(array)

    print(f"Saved CSV to: {csv_file}")
    #"""
# Example usage
if __name__ == "__main__":
    npy_file = "rl/datasets/minimal_viable_v8/demo_states_0.npy" 
    npy_to_csv(npy_file)

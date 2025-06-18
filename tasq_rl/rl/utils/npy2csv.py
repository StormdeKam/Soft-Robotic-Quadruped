import numpy as np
import csv
import os

def npy_to_csv(npy_file, csv_file=None):
    # Load the NumPy array
    array = np.load(npy_file, allow_pickle=False)
    #print(array.shape)
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
        writer.writerows(array[:100,:])

    print(f"Saved CSV to: {csv_file}")
    #"""
# Example usage
if __name__ == "__main__":
    npy_file = "rl/datasets/RNN/demo_actions.npy"  # <-- change this to your file path
    npy_to_csv(npy_file)

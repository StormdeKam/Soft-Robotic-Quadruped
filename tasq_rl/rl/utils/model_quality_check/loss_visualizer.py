import matplotlib.pyplot as plt
import csv

# File path to the log data
loss_data_file = "rl/models/tasq_bc/minimal_viable_V7/tasq_bc_log.txt"

# Read data from file
epochs = []
train_total, train_pressure, train_suction = [], [], []
val_total, val_pressure, val_suction = [], [], []

with open(loss_data_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header line
    for row in reader:
        epochs.append(int(row[0]))
        train_total.append(float(row[1]))
        train_pressure.append(float(row[2]))
        train_suction.append(float(row[3]))
        val_total.append(float(row[4]))
        val_pressure.append(float(row[5]))
        val_suction.append(float(row[6]))

# Plot all in one figure with 3 subplots
plt.figure(figsize=(12, 10))

# --- Total Loss ---
plt.subplot(3, 1, 1)
plt.plot(epochs, train_total, label='Train Total Loss', marker='o')
plt.plot(epochs, val_total, label='Validation Total Loss', marker='s')
plt.yscale('log')  # <--- Log scale
plt.ylabel('Loss')
plt.title('Total Loss')
plt.legend()
plt.grid(True)

# --- Pressure Loss ---
plt.subplot(3, 1, 2)
plt.plot(epochs, train_pressure, label='Train Pressure Loss', marker='o')
plt.plot(epochs, val_pressure, label='Validation Pressure Loss', marker='s')
plt.yscale('log')  # <--- Log scale
plt.ylabel('Loss')
plt.title('Pressure Loss')
plt.legend()
plt.grid(True)

# --- Suction Loss ---
plt.subplot(3, 1, 3)
plt.plot(epochs, train_suction, label='Train Suction Loss', marker='o')
plt.plot(epochs, val_suction, label='Validation Suction Loss', marker='s')
plt.yscale('log')  # <--- Log scale
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Suction Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/loss_4_8.png", dpi=300)
plt.show()

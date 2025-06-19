import matplotlib.pyplot as plt
import csv

# File path to the log data
loss_data_file = "rl/models/tasq_bc/no_steady_states/tasq_bc_log.txt"

# Read data from file
data = []
with open(loss_data_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header line
    for row in reader:
        epoch = int(row[0])
        train_loss = float(row[1])
        val_loss = float(row[2])
        data.append((epoch, train_loss, val_loss))

# Split data into components
epochs = [row[0] for row in data]
train_losses = [row[1] for row in data]
val_losses = [row[2] for row in data]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image or show it
plt.savefig("loss_plot.png", dpi=300)
plt.show()


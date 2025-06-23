import numpy as np
import matplotlib.pyplot as plt
import csv
from rl.scripts.tasq_collect_dataset import demo_pressures

def load_log_data(filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            converted_row = []
            for x in row:
                if x == "True":
                    converted_row.append(1.0)
                elif x == "False":
                    converted_row.append(0.0)
                else:
                    converted_row.append(float(x))
            data.append(converted_row)
    return header, np.array(data)

def load_true_data(timesteps):
    demo_pressure_states = []
    demo_pressure_state = np.zeros(12)
    for step in timesteps:
        demo_pressure_state += demo_pressures(step).flatten()
        demo_pressure_states.append(demo_pressure_state.copy())
    return np.array(demo_pressure_states)

def plot_pressures_and_suctions(filepath):
    header, data = load_log_data(filepath)
    timesteps = np.arange(len(data))

    demo_pressure_states = load_true_data(timesteps)
    
    pressures = data[:, :12]  # first 12 columns
    
    # --- Plot true pressures ---
    plt.figure(figsize=(12, 6))
    for i in range(12):
        plt.plot(timesteps, demo_pressure_states[:, i], label=f'P{i+1}')
    plt.title("Target pressures Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Pressure (mBar)")
    plt.legend(loc='upper right', ncol=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/pressures_plot.png")
    plt.show()

    # --- Plot BC pressures ---
    plt.figure(figsize=(12, 6))
    for i in range(12):
        plt.plot(timesteps, pressures[:, i], label=f'P{i+1}')
    plt.title("BC model pressures Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Pressure (mBar)")
    plt.legend(loc='upper right', ncol=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/pressures_plot.png")
    plt.show()

    # --- Plot pressure errors ---
    plt.figure(figsize=(12, 6))
    for i in range(12):
        errors = pressures - demo_pressure_states
        plt.plot(timesteps, errors[:, i], label=f'P{i+1}')
    plt.title("Pressure State Error Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Pressure error(mBar)")
    plt.legend(loc='upper right', ncol=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/error_plot.png")
    plt.show()

# --- Call the function with your CSV file ---
if __name__ == "__main__":
    plot_pressures_and_suctions("rl/log/tasq_log.csv")

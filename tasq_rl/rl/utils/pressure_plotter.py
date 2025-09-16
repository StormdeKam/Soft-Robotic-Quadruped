import numpy as np
import matplotlib.pyplot as plt
import csv

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

def plot_pressures_and_suctions(filepath):
    header, data = load_log_data(filepath)
    
    pressures = data[:, :12]  # first 12 columns
    suctions = data[:, 12:]  # last 4 columns
    
    timesteps = np.arange(len(data))
    
    # --- Plot pressures ---
    plt.figure(figsize=(12, 6))
    for i in range(12):
        plt.plot(timesteps, pressures[:, i], label=f'P{i+1}')
    plt.title("Pressures Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Pressure (mBar)")
    plt.legend(loc='upper right', ncol=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/pressures_plot.png")
    plt.show()

    # --- Plot suction states ---
    plt.figure(figsize=(10, 4))
    for i in range(4):
        plt.plot(timesteps, suctions[:, i], label=f'Suction {i+1}')
    plt.title("Suction Cup States Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Suction State")
    plt.ylim(-10, 110)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/suction_states_plot.png")
    plt.show()

# --- Call the function with your CSV file ---
if __name__ == "__main__":
    plot_pressures_and_suctions("rl/log/tasq_log.csv")

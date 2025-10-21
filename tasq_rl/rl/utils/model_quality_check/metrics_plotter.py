import re
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
BL = 0.2  # body length in meters
EPISODE_STEPS = 7000
CONTROL_FREQ = 24.0  # Hz

# --- Parse log file ---
def parse_metrics_log(file_path):
    steps = []
    avg_final_x = []

    with open(file_path, "r") as f:
        for line in f:
            match = re.match(r"Steps (\d+):.*'avg_final_x': ([0-9.eE+-]+)", line)
            if match:
                steps.append(int(match.group(1)))
                avg_final_x.append(float(match.group(2)))

    return np.array(steps), np.array(avg_final_x)


# --- Compute speed ---
def compute_speed(avg_final_x):
    time_per_episode = EPISODE_STEPS / CONTROL_FREQ
    return (avg_final_x / BL) / time_per_episode  # BL/s


# --- Main ---
def main():
    file_path = "rl/models/tasq_sac/low_freq/trial_3(10_10_2025)/actors/metrics.txt"
    steps, avg_final_x = parse_metrics_log(file_path)

    if len(steps) == 0:
        print("❌ No data found in metrics.txt")
        return

    episodes = steps / (EPISODE_STEPS/100)
    speed = compute_speed(avg_final_x)

    # --- Fit polynomial trendline ---
    degree = 1  # You can change this to 2 for a simpler curve
    coeffs = np.polyfit(episodes, speed, degree)
    poly = np.poly1d(coeffs)
    trend_x = np.linspace(episodes.min(), episodes.max(), 200)
    trend_y = poly(trend_x)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(episodes, speed, color="tab:blue", alpha=0.3, label="Measured Speed (BL/s)")
    plt.plot(trend_x, trend_y, color="tab:blue", linewidth=1)
    plt.ylim(0, 0.0015)
    plt.title("Speed over Training Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Speed (BL/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_log(csv_path="reward_log.csv", window=50, save_fig=False):
    # Load the CSV
    df = pd.read_csv(csv_path)
    rewards = df["reward"].values
    lengths = df["length"].values
    episodes = range(1, len(rewards) + 1)

    # Compute moving average
    if window > 1:
        smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    else:
        smoothed = rewards

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.4)
    plt.plot(episodes, smoothed, label=f"Smoothed Reward (window={window})", linewidth=2)

    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    if save_fig:
        plt.savefig("reward_plot.png")
        print("[✓] Saved plot to reward_plot.png")
    else:
        plt.show()

if __name__ == "__main__":
    plot_reward_log("reward_log.csv", window=50, save_fig=False)

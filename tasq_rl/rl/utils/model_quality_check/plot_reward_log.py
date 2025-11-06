import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_log(csv_file="reward_log.csv", rolling=10):
    # Define correct column names (matches your CSV content)
    cols = [
        "reward", "length", "alive", "forward",
        "orientation_penalty", "height_penalty",
        "pressure_smoothness", "suction_smoothness", "total_forward", "?"
    ]

    # Load CSV using explicit column names
    df = pd.read_csv(csv_file, names=cols, header=0)
    df = df.iloc[83:]

    # Separate reward components (all columns except total reward and length)
    component_cols = [c for c in df.columns if c not in ["reward", "length"]]

    # Apply rolling average (smooth noisy curves)
    df_smooth = df.rolling(window=rolling, min_periods=1).mean() if rolling > 1 else df

    # --- Plot total reward ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_smooth["reward"], label="Total Smoothed Reward")
    plt.plot(df["reward"], label="Total Reward ", color="blue", alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_total.png", transparent=True, dpi=300)
    plt.show()

    # --- Plot episode length ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_smooth["length"], label="Episode Length", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Length per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_length.png", transparent=True, dpi=300)
    plt.show()

    # --- Plot individual reward components ---
    if component_cols:
        plt.figure(figsize=(10, 6))
        for c in component_cols:
            plt.plot(df[c], label=c)
        plt.xlabel("Episode")
        plt.ylabel("Reward Contribution")
        plt.title("Reward Components")
        plt.legend()
        plt.grid(True)
        plt.savefig("reward_components.png", transparent=True, dpi=300)
        plt.show()

if __name__ == "__main__":
    plot_reward_log("rl/models/tasq_sac/low_freqv6/sac3/trial7_05_11_25/reward_log.csv", rolling=200)

import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_log(csv_file="reward_log.csv", rolling=10):
    # Load CSV
    col_names = ["reward", "length", "survival", "goal distance", "rel goal orientation", "height deviation", "pitch/roll deviation", "pressure smoothness", "suction smoothness"]
    df = pd.read_csv(csv_file, names=col_names, header=0)

    # Ensure all numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Separate reward components (all columns except reward/length)
    component_cols = [c for c in df.columns if c not in ["reward", "length"]]

    # Rolling average (to smooth noisy curves)
    if rolling > 1:
        df_smooth = df.rolling(rolling, min_periods=1).mean()
    else:
        df_smooth = df

    # Plot total reward
    plt.figure(figsize=(10,6))
    plt.plot(df_smooth["reward"], label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot episode length
    plt.figure(figsize=(10,6))
    plt.plot(df_smooth["length"], label="Episode Length", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Length per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot each reward component
    if component_cols:
        plt.figure(figsize=(10,6))
        plt.plot(df_smooth["reward"], label="Total Reward")
        for c in component_cols:
            plt.plot(df_smooth[c], label=c)
        plt.xlabel("Episode")
        plt.ylabel("Reward Contribution")
        plt.title("Reward Components")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    plot_reward_log("rl/models/tasq_sac/minimal_viable_v3/trial_7(6_10_2025)/reward_log.csv", rolling=10)

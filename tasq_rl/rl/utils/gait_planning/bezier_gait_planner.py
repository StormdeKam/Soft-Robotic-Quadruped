import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import csv

class BezierGaitPlanner:
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.points = [[] for _ in range(num_channels)]  # Each: list of (t, value)

    def add_point(self, channel, time, value):
        assert 0 <= channel < self.num_channels, "Invalid channel index"
        self.points[channel].append((time, value))
        self.points[channel].sort()  # Keep sorted by time

    def evaluate(self, t):
        results = np.zeros(self.num_channels)
        for i, pts in enumerate(self.points):
            if len(pts) < 2:
                results[i] = 0
                continue
            times, values = zip(*pts)
            times = np.array(times)
            values = np.array(values)
            if t <= times[0]:
                results[i] = values[0]
            elif t >= times[-1]:
                results[i] = values[-1]
            else:
                spline = make_interp_spline(times, values, k=min(3, len(pts) - 1))
                results[i] = spline(t)
        return results

    def get_gradients_at_time(self, n, dt=1/24):
        """
        Computes the gradient of each channel at a specific step n using finite difference:
        (evaluate(t + dt) - evaluate(t)) / dt
        """
        t = n*dt
        p1 = self.evaluate(t)
        p2 = self.evaluate(t + dt)
        gradient = p2 - p1
        return gradient
    
    def plot(self, duration=1.0, dt=0.01):
        time = np.arange(0, duration, dt)
        signals = np.array([self.evaluate(t) for t in time])

        plt.figure(figsize=(12, 8))
        for i in range(self.num_channels):
            plt.plot(time, signals[:, i], label=f'Channel {i}')
            # Also show key points
            pts = self.points[i]
            if pts:
                pt_times, pt_vals = zip(*pts)
                plt.plot(pt_times, pt_vals, 'o', label=f'Points {i}')
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.title("Bezier Gait Planner")
        plt.grid(True)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.show()

    def save_to_csv(self, filename="gait_data.csv", duration=1.0, dt=0.01):
        """
        Save the pressure/suction values for each channel over time into a CSV file.

        Args:
            filename (str): Output CSV file name
            duration (float): Total duration to simulate
            dt (float): Sampling timestep
        """
        time = np.arange(0, duration, dt)
        signals = np.array([self.evaluate(t) for t in time])

        header = ["time"] + [f"channel_{i}" for i in range(self.num_channels)]

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t, row in zip(time, signals):
                writer.writerow([t] + row.tolist())

        print(f"Saved gait data to {filename}")

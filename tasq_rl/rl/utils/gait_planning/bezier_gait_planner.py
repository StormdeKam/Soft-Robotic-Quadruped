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

        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.get_cmap("tab20")
        lines = []

        for i in range(self.num_channels):
            color = cmap(i % 20)
            # Main line
            line, = ax.plot(time, signals[:, i], label=f'Channel {i}', color=color)
            lines.append(line)

            # Key points
            pts = self.points[i]
            if pts:
                pt_times, pt_vals = zip(*pts)
                pt_line, = ax.plot(pt_times, pt_vals, 'o', color=color, markersize=5, label=f'Points {i}')
                lines.append(pt_line)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.set_title("Bezier Gait Planner")
        ax.grid(True)

        # Legend
        leg = ax.legend(ncol=2)

        # Map legend entries to original lines
        legend_map = {}
        for legline, origline in zip(leg.legend_handles, lines):
            legline.set_picker(5)  # 5 pts tolerance
            legend_map[legline] = origline
        # Also map the legend text so it's clickable too
        for text, origline in zip(leg.get_texts(), lines):
            text.set_picker(True)
            legend_map[text] = origline

        def on_pick(event):
            leg_item = event.artist
            if leg_item not in legend_map:
                return
            orig_line = legend_map[leg_item]
            visible = not orig_line.get_visible()
            orig_line.set_visible(visible)

            # Fade legend entry
            for k, v in legend_map.items():
                if v is orig_line:
                    if isinstance(k, plt.Text):
                        k.set_alpha(1.0 if visible else 0.4)
                    else:
                        k.set_alpha(1.0 if visible else 0.2)

            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)
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

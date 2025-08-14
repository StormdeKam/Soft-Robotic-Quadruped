import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline

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
        t = n * dt
        p1 = self.evaluate(t)
        p2 = self.evaluate(t + dt)
        gradient = p2 - p1
        return gradient

    def plot(self, duration=1.0, dt=0.01):
        import plotly.graph_objects as go

        time = np.arange(0, duration, dt)
        signals = np.array([self.evaluate(t) for t in time])

        fig = go.Figure()

        for i in range(self.num_channels):
            # Plot spline curve
            fig.add_trace(go.Scatter(
                x=time,
                y=signals[:, i],
                mode='lines',
                name=f'Channel {i}',
                line=dict(width=2)
            ))

            # Plot control points
            pts = self.points[i]
            if pts:
                pt_times, pt_vals = zip(*pts)
                fig.add_trace(go.Scatter(
                    x=pt_times,
                    y=pt_vals,
                    mode='markers',
                    name=f'Points {i}',
                    marker=dict(size=8, symbol='circle-open')
                ))

        fig.update_layout(
            title="Bezier Gait Planner (Interactive)",
            xaxis_title="Time (s)",
            yaxis_title="Value",
            legend_title="Legend",
            hovermode="x unified",
            template="plotly_white",
            width=1000,
            height=600
        )

        fig.show()

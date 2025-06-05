import matplotlib.pyplot as plt
import numpy as np

class PressurePlotter:
    def __init__(self, steps):
        self.steps = steps
        self.pressure_history = np.zeros((12, steps))
        self.x_data = np.arange(steps)

        plt.ion()
        self.fig, self.axes = plt.subplots(nrows=12, ncols=1, figsize=(8, 16), sharex=True)

        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot(self.x_data, np.zeros_like(self.x_data), label=f'P[{i}]', color='tab:blue')
            ax.set_ylabel(f'P[{i}]')
            ax.set_ylim(0, 500)  # adjust range as needed
            ax.grid(True)
            self.lines.append(line)

        self.axes[-1].set_xlabel("Step")
        self.fig.tight_layout()

        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for ax in self.axes:
            ax.draw_artist(ax.patch)
            for line in ax.lines:
                ax.draw_artist(line)
        self.fig.canvas.blit(self.fig.bbox)

    def plot_pressures(self, step, flat_pressures):
        self.pressure_history[:, step] = flat_pressures

        for i, line in enumerate(self.lines):
            line.set_ydata(self.pressure_history[i])
            self.axes[i].draw_artist(line)

        self.fig.canvas.restore_region(self.bg)
        for ax in self.axes:
            for line in ax.lines:
                ax.draw_artist(line)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def finish(self):
        plt.ioff()
        plt.show()

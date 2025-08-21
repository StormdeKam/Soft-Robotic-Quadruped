import sys
import json
import copy
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout,
    QWidget, QHBoxLayout, QCheckBox, QScrollArea, QShortcut
)
from PyQt5.QtGui import QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt


class GaitEditor(FigureCanvas):
    def __init__(self, num_channels=16, duration=10.0, dt=0.01):
        self.fig = Figure(figsize=(12, 6))
        super().__init__(self.fig)

        self.num_channels = num_channels
        self.duration = duration
        self.dt = dt
        self.time_array = np.arange(0, duration, dt)
        self.channel_visible = [True] * num_channels
        self.channel_editable = [False] * num_channels  # New: edit mode per channel

        self.hovered_points = set()

        # Start with 1 point at time=0, value=0 per channel
        self.points = [
            [(0.0, 0.0)] for _ in range(num_channels)
        ]

        self.history = []
        self.dragging = None  # Will store dict {channel_idx: point_idx} when dragging multiple
        self.hovered_point = None  # (channel_idx, point_idx)
        self.ax = self.fig.add_subplot(111)

        # Connect matplotlib events
        self.cid_press = self.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.mpl_connect("motion_notify_event", self.on_motion)
        #self.cid_pick = self.mpl_connect("motion_notify_event", self.on_hover)

        self.draw_plot()

    def export_gait(self):
        return {
            f"channel_{i}": self.points[i]
            for i in range(self.num_channels)
        }

    def save_history(self):
        self.history.append(copy.deepcopy(self.points))

    def undo(self):
        if not self.history:
            return
        self.points = self.history.pop()
        self.draw_plot()

    def set_channel_visibility(self, channel, visible):
        self.channel_visible[channel] = visible
        self.draw_plot()

    def set_channel_editable(self, channel, editable):
        self.channel_editable[channel] = editable
        self.draw_plot()

    def draw_plot(self):
        self.ax.clear()
        self.handles = []

        for i in range(self.num_channels):
            if not self.channel_visible[i]:
                continue

            pts = self.points[i]
            if len(pts) == 0:
                continue

            t_vals, y_vals = zip(*pts)

            # Draw lines between points (no spline)
            self.ax.plot(t_vals, y_vals, '-', label=f"Ch {i}")

            # Draw points, color red if hovered else blue
            for j, (t, v) in enumerate(pts):
                if (i, j) in getattr(self, 'hovered_points', set()):
                    color = 'red'
                else:
                    color = 'blue'
                scatter = self.ax.plot(t, v, 'o', color=color, picker=5)[0]
                self.handles.append(scatter)

        self.ax.set_xlim(0, self.duration)
        self.ax.set_ylim(0, 400)
        self.ax.set_title("Multi-Channel Gait Editor")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pressure (mbar)")
        self.ax.legend(loc="upper right", fontsize=8, ncol=4)
        self.draw()

    def update_hover(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            if getattr(self, 'hovered_points', None):
                self.hovered_points = set()
                self.setCursor(Qt.ArrowCursor)
                self.draw_plot()
            return

        hover_radius_time = 0.2  # smaller radius for more precision
        hover_radius_val = 5

        new_hovered = set()
        for i in range(self.num_channels):
            if not self.channel_visible[i]:
                continue
            pts = self.points[i]
            for j, (t, v) in enumerate(pts):
                if abs(event.xdata - t) < hover_radius_time and abs(event.ydata - v) < hover_radius_val:
                    new_hovered.add((i, j))

        if new_hovered != getattr(self, 'hovered_points', set()):
            self.hovered_points = new_hovered
            self.draw_plot()

        if self.hovered_points:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def on_motion(self, event):
        # First update hovering state
        self.update_hover(event)

        # Then handle dragging if active
        if self.dragging is None:
            return

        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        for (i, j) in self.dragging:
            new_t = float(np.clip(event.xdata, 0, self.duration))
            new_v = float(np.clip(event.ydata, 0, 400))

            other_times = [self.points[i][k][0] for k in range(len(self.points[i])) if k != j]
            if any(abs(new_t - t) < 1e-4 for t in other_times):
                new_t += 1e-5

            self.points[i][j] = (new_t, new_v)
            self.points[i] = sorted(self.points[i], key=lambda p: p[0])

        self.draw_plot()

    def on_press(self, event):
        if event.inaxes is None:
            return

        # Look for points near cursor in editable channels
        hover_radius = 0.2  # tweak if needed
        dragging_points = []

        for i in range(self.num_channels):
            if not self.channel_visible[i] or not self.channel_editable[i]:
                continue
            pts = self.points[i]
            for j, (t, v) in enumerate(pts):
                dist = np.hypot(event.xdata - t, event.ydata - v)
                if dist < hover_radius:
                    dragging_points.append((i, j))

        if dragging_points:
            self.save_history()
            self.dragging = dragging_points
            return  # DO NOT add points

        # No points nearby => add new points to all editable channels
        edit_channels = [idx for idx, editable in enumerate(self.channel_editable) if editable]
        if not edit_channels:
            # fallback to first visible channel only
            edit_channels = [i for i, vis in enumerate(self.channel_visible) if vis]

        if not edit_channels:
            return  # no editable or visible channels, do nothing

        self.save_history()
        new_t = float(np.clip(event.xdata, 0, self.duration))
        new_v = float(np.clip(event.ydata, 0, 400))
        for i in edit_channels:
            self.points[i].append((new_t, new_v))
            self.points[i] = sorted(self.points[i], key=lambda p: p[0])
        self.draw_plot()


    def on_release(self, event):
        self.dragging = None

    def find_nearby_points(self, event, threshold=5):
        """Return dict of channel_idx: point_idx for points near event."""
        if event.inaxes is None:
            return {}

        near_points = {}
        for i in range(self.num_channels):
            if not self.channel_visible[i] or not self.channel_editable[i]:
                continue
            pts = self.points[i]
            for j, (t, v) in enumerate(pts):
                dist = np.hypot(event.xdata - t, event.ydata - v)
                if dist < threshold:
                    near_points[i] = j
                    break
        return near_points
    
    def load_gait(self, data):
        for i in range(self.num_channels):
            loaded_pts = data.get(f"channel_{i}", [])
            self.points[i] = [(float(t), float(v)) for t, v in loaded_pts]
        self.draw_plot()


class GaitEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Channel Gait Editor")

        self.editor = GaitEditor()

        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.editor.undo)

        # Channel visibility checkboxes
        checkbox_layout = QVBoxLayout()
        self.visibility_checkboxes = []
        self.edit_checkboxes = []
        self.editor.edit_checkboxes = self.edit_checkboxes  # after you create self.edit_checkboxes list


        for i in range(self.editor.num_channels):
            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)

            cb_vis = QCheckBox(f"Show Ch {i}")
            cb_vis.setChecked(True)
            cb_vis.stateChanged.connect(self.make_visibility_callback(i))
            self.visibility_checkboxes.append(cb_vis)

            cb_edit = QCheckBox("Edit")
            cb_edit.setChecked(False)
            cb_edit.stateChanged.connect(self.make_edit_callback(i))
            self.edit_checkboxes.append(cb_edit)

            row_layout.addWidget(cb_vis)
            row_layout.addWidget(cb_edit)
            row_widget.setLayout(row_layout)

            checkbox_layout.addWidget(row_widget)

        checkbox_container = QWidget()
        checkbox_container.setLayout(checkbox_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(checkbox_container)
        scroll.setFixedWidth(200)

        save_btn = QPushButton("Save Gait")
        load_btn = QPushButton("Load Gait")
        save_btn.clicked.connect(self.save_gait)
        load_btn.clicked.connect(self.load_gait)

        button_layout = QHBoxLayout()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(load_btn)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.editor)

        right_panel = QVBoxLayout()
        right_panel.addWidget(scroll)
        right_panel.addLayout(button_layout)

        main_layout.addLayout(right_panel)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def make_visibility_callback(self, channel):
        def callback(state):
            visible = state == 2
            self.editor.set_channel_visibility(channel, visible)
        return callback

    def make_edit_callback(self, channel):
        def callback(state):
            editable = state == 2
            self.editor.set_channel_editable(channel, editable)
        return callback

    def save_gait(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Gait", "", "JSON Files (*.json)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            with open(path, "w") as f:
                json.dump(self.editor.export_gait(), f, indent=2)

    def load_gait(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Gait", "", "JSON Files (*.json)")
        if path:
            with open(path, "r") as f:
                data = json.load(f)
                self.editor.load_gait(data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GaitEditorWindow()
    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec_())

#!/usr/bin/env python3
"""
Ring Resonator Design Studio - Enhanced GUI Implementation
Interactive GUI with dark theme matplotlib plots and comprehensive ring visualization
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QSplitter,
    QFrame,
    QScrollArea,
    QToolBar,
    QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, QThread, QObject, Signal
from PySide6.QtGui import QFont, QAction

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.style as mplstyle

# Configure matplotlib for dark theme
plt.style.use("dark_background")
mplstyle.use("dark_background")

# Import ring resonator modules
try:
    from .rings_design import RingResonatorSystem, RingAnalyzer
except ImportError:
    from rings_design import RingResonatorSystem, RingAnalyzer


class MplCanvas(FigureCanvas):
    """Enhanced matplotlib canvas widget with dark theme"""

    def __init__(self, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#2e2e2e")
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(None)

        # Set dark theme colors
        self.fig.patch.set_facecolor("#2e2e2e")
        self.axes.set_facecolor("#2e2e2e")
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")
        self.axes.tick_params(colors="white")
        self.axes.title.set_color("white")

        # Enable grid with dark theme
        self.axes.grid(True, alpha=0.3, color="gray")

        # Enable tight layout to maximize plot area but prevent cut-off
        self.fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98], pad=0.2)

    def clear_plot(self):
        """Clear the plot while maintaining dark theme"""
        self.axes.clear()
        self.axes.set_facecolor("#2e2e2e")
        self.axes.xaxis.label.set_color("white")
        self.axes.yaxis.label.set_color("white")
        self.axes.tick_params(colors="white")
        self.axes.title.set_color("white")
        self.axes.grid(True, alpha=0.3, color="gray")
        self.draw()


class InteractiveGeometryPlot(QWidget):
    """Enhanced ring geometry visualization with detailed annotations"""

    def __init__(self):
        super().__init__()
        self.log_callback = None
        self.setup_ui()
        self.ring_radii = []
        self.coupling_coeffs = []
        self.phase_shifts = []

    def safe_log(self, message):
        """Safely log a message if callback is available"""
        if self.log_callback:
            self.log_callback(message)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Control toolbar
        toolbar = QToolBar()
        toolbar.setStyleSheet("QToolBar { background-color: #404040; border: 1px solid #555; }")

        # Show/hide options
        self.show_labels = QCheckBox("Show Labels")
        self.show_labels.setChecked(True)
        self.show_labels.stateChanged.connect(self.refresh_plot)
        self.show_labels.stateChanged.connect(
            lambda state: self.log_callback
            and self.log_callback(f"Labels display {'enabled' if state else 'disabled'}")
        )
        toolbar.addWidget(self.show_labels)

        self.show_dimensions = QCheckBox("Show Dimensions")
        self.show_dimensions.setChecked(True)
        self.show_dimensions.stateChanged.connect(self.refresh_plot)
        self.show_dimensions.stateChanged.connect(
            lambda state: self.log_callback
            and self.log_callback(f"Dimensions display {'enabled' if state else 'disabled'}")
        )
        toolbar.addWidget(self.show_dimensions)

        self.show_coupling = QCheckBox("Show Coupling")
        self.show_coupling.setChecked(True)
        self.show_coupling.stateChanged.connect(self.refresh_plot)
        self.show_coupling.stateChanged.connect(
            lambda state: self.log_callback
            and self.log_callback(f"Coupling display {'enabled' if state else 'disabled'}")
        )
        toolbar.addWidget(self.show_coupling)

        layout.addWidget(toolbar)

        # Create matplotlib canvas with navigation - increased size for better visibility
        self.canvas = MplCanvas(width=16, height=10)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.nav_toolbar.setStyleSheet("QToolBar { background-color: #404040; }")

        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.show_placeholder()

    def show_placeholder(self):
        """Show enhanced placeholder text"""
        self.canvas.clear_plot()
        self.canvas.axes.text(
            0.5,
            0.5,
            "Configure Ring Parameters\nand See Live Geometry Preview",
            transform=self.canvas.axes.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#888",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#404040", alpha=0.8),
        )
        # Remove title to save vertical space
        self.canvas.draw()

    def refresh_plot(self):
        """Refresh the plot with current settings"""
        self.safe_log("Geometry plot refreshed due to display option change")
        if self.ring_radii:
            self.update_geometry(self.ring_radii, self.coupling_coeffs, self.phase_shifts)

    def update_geometry(
        self,
        ring_radii: List[float],
        coupling_coeffs: List[float],
        phase_shifts: List[float] = None,
    ):
        """Update the enhanced geometry plot with detailed annotations"""
        if not ring_radii:
            self.show_placeholder()
            return

        self.ring_radii = ring_radii
        self.coupling_coeffs = coupling_coeffs
        self.phase_shifts = phase_shifts or [0.0] * len(ring_radii)

        # Clear plot
        self.canvas.clear_plot()

        # Calculate enhanced layout
        max_radius = max(ring_radii)
        spacing = max_radius * 2.8  # Increased spacing for better visibility

        # Ring positions with optimized layout
        ring_positions = []
        for i, radius in enumerate(ring_radii):
            y_pos = -i * spacing
            ring_positions.append((0, y_pos))

        # Bus waveguide positions
        bus_top_y = ring_positions[0][1] + max_radius + 40
        bus_bottom_y = ring_positions[-1][1] - max_radius - 40

        # Extended plot extent for maximum width spanning - even longer to fill all space
        x_extent = max_radius * 18.0  # Increased to 18.0 for maximum width spanning

        # Draw enhanced bus waveguides with extended length
        self.canvas.axes.plot(
            [-x_extent, x_extent], [bus_top_y, bus_top_y], "#4CAF50", linewidth=6, alpha=0.8
        )
        self.canvas.axes.plot(
            [-x_extent, x_extent], [bus_bottom_y, bus_bottom_y], "#F44336", linewidth=6, alpha=0.8
        )

        # Draw rings with enhanced styling
        ring_colors = [
            "#2196F3",
            "#4CAF50",
            "#FF9800",
            "#9C27B0",
            "#F44336",
            "#00BCD4",
            "#795548",
            "#607D8B",
        ]

        for i, (radius, (x, y)) in enumerate(zip(ring_radii, ring_positions)):
            color = ring_colors[i % len(ring_colors)]

            # Main ring circle
            circle = patches.Circle((x, y), radius, fill=False, color=color, linewidth=3, alpha=0.9)
            self.canvas.axes.add_patch(circle)

            # Ring center point
            self.canvas.axes.plot(x, y, "o", color=color, markersize=8, alpha=0.8)

            # Show dimensions if enabled
            if self.show_dimensions.isChecked():
                # Radius value (without arrow)
                self.canvas.axes.text(
                    x + radius / 2,
                    y + 8,
                    f"R={radius:.1f}μm",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#404040", alpha=0.8),
                )

            # Phase shift annotation
            if self.phase_shifts[i] != 0:
                phase_text = f"φ={self.phase_shifts[i]:.2f}π"
                self.canvas.axes.text(
                    x + radius + 15,
                    y,
                    phase_text,
                    ha="left",
                    va="center",
                    fontsize=10,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#404040", alpha=0.8),
                )

        # Add port labels with enhanced styling (inside plot area, closer to center)
        if self.show_labels.isChecked():
            # Position labels closer to center (0.6 instead of 0.8)
            label_x_offset = x_extent * 0.6

            # Input port (inside plot, left side)
            self.canvas.axes.text(
                -label_x_offset,
                bus_top_y,
                "INPUT",
                ha="center",
                va="center",
                fontsize=12,
                color="#4CAF50",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2e2e2e", edgecolor="#4CAF50"),
            )

            # Through port (inside plot, right side)
            self.canvas.axes.text(
                label_x_offset,
                bus_top_y,
                "THROUGH",
                ha="center",
                va="center",
                fontsize=12,
                color="#4CAF50",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2e2e2e", edgecolor="#4CAF50"),
            )

            # Drop port position depends on ring count (odd=left, even=right)
            num_rings = len(ring_radii)
            if num_rings % 2 == 1:  # Odd number of rings - drop on left
                drop_x = -label_x_offset
            else:  # Even number of rings - drop on right
                drop_x = label_x_offset

            self.canvas.axes.text(
                drop_x,
                bus_bottom_y,
                "DROP",
                ha="center",
                va="center",
                fontsize=12,
                color="#F44336",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2e2e2e", edgecolor="#F44336"),
            )

        # Add coupling coefficient annotations (all in middle area)
        if self.show_coupling.isChecked():
            for i, k in enumerate(coupling_coeffs):
                coupling_color = "#FFC107"
                if i == 0:
                    # Input coupling (moved to middle)
                    y_pos = ring_positions[0][1] + ring_radii[0] + 20
                    self.canvas.axes.text(
                        0,
                        y_pos,
                        f"κ₁ = {k:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        color=coupling_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#404040", alpha=0.9),
                    )
                elif i == len(coupling_coeffs) - 1:
                    # Output coupling (moved to middle)
                    y_pos = ring_positions[-1][1] - ring_radii[-1] - 20
                    self.canvas.axes.text(
                        0,
                        y_pos,
                        f"κ{i+1} = {k:.3f}",
                        ha="center",
                        va="top",
                        fontsize=11,
                        color=coupling_color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#404040", alpha=0.9),
                    )
                else:
                    # Inter-ring coupling (moved to middle)
                    if i < len(ring_positions):
                        y_pos = (ring_positions[i - 1][1] + ring_positions[i][1]) / 2
                        self.canvas.axes.text(
                            0,
                            y_pos,
                            f"κ{i+1} = {k:.3f}",
                            ha="center",
                            va="center",
                            fontsize=11,
                            color=coupling_color,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#404040", alpha=0.9),
                        )

        # Enhanced plot styling - fill available space
        self.canvas.axes.set_aspect("equal")

        # Calculate optimal limits to maintain FIXED consistent width regardless of ring count
        total_height = abs(bus_bottom_y - bus_top_y) + 80  # Height based on ring layout

        # Use FIXED width for all ring counts to ensure perfect consistency
        # This ensures rings=1 width exactly matches rings>1 width
        bus_based_width = x_extent * 2.1

        # Use a FIXED width that's the same for all ring counts
        # This is the key to preventing width variation
        FIXED_CONSISTENT_WIDTH = 1400  # Fixed width for all ring counts

        # Always use the fixed width to ensure absolute consistency
        total_width = max(bus_based_width, FIXED_CONSISTENT_WIDTH)

        # Center the plot and expand to fill space
        center_y = (bus_top_y + bus_bottom_y) / 2
        self.canvas.axes.set_xlim(-total_width / 2, total_width / 2)
        self.canvas.axes.set_ylim(center_y - total_height / 2, center_y + total_height / 2)

        # Remove axis labels for cleaner appearance
        self.canvas.axes.set_xlabel("")
        self.canvas.axes.set_ylabel("")
        # Remove title to save vertical space for better plot visibility

        # Legend removed for cleaner appearance - not helpful for ring visualization

        # Tight layout to maximize plot area but prevent cut-off
        self.canvas.fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98], pad=0.3)
        self.canvas.draw()

        print(f"Enhanced geometry updated: {len(ring_radii)} rings plotted with full annotations")


class InteractiveAnalysisPlot(QWidget):
    """Enhanced analysis results visualization with dark theme and interactivity"""

    def __init__(self):
        super().__init__()
        self.log_callback = None
        self.setup_ui()
        self.current_results = None

    def safe_log(self, message):
        """Safely log a message if callback is available"""
        if self.log_callback:
            self.log_callback(message)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Enhanced control panel
        control_panel = QFrame()
        control_panel.setStyleSheet(
            "QFrame { background-color: #404040; border: 1px solid #555; border-radius: 5px; padding: 5px; }"
        )
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Analysis Type:"))
        self.plot_selector = QComboBox()
        self.plot_selector.addItems(
            [
                "Transmission Spectrum",
                "Phase Response",
                "Group Delay",
                "Dispersion",
                "Effective Index",
            ]
        )
        self.plot_selector.currentTextChanged.connect(self.update_plot)
        self.plot_selector.setStyleSheet(
            "QComboBox { background-color: #555; color: white; border: 1px solid #777; }"
        )
        control_layout.addWidget(self.plot_selector)

        control_layout.addStretch()

        # Plot options
        self.show_grid = QCheckBox("Grid")
        self.show_grid.setChecked(True)
        self.show_grid.stateChanged.connect(self.update_plot)
        self.show_grid.stateChanged.connect(
            lambda state: self.log_callback
            and self.log_callback(f"Grid display {'enabled' if state else 'disabled'}")
        )
        control_layout.addWidget(self.show_grid)

        self.show_markers = QCheckBox("Markers")
        self.show_markers.setChecked(False)
        self.show_markers.stateChanged.connect(self.update_plot)
        self.show_markers.stateChanged.connect(
            lambda state: self.log_callback
            and self.log_callback(f"Markers display {'enabled' if state else 'disabled'}")
        )
        control_layout.addWidget(self.show_markers)

        self.export_btn = QPushButton("Export Plot")
        self.export_btn.clicked.connect(self.export_plot)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; border: none; padding: 5px 10px; border-radius: 3px; }"
        )
        control_layout.addWidget(self.export_btn)

        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

        # Create matplotlib canvas with navigation - increased size for better visibility
        self.canvas = MplCanvas(width=16, height=8)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.nav_toolbar.setStyleSheet("QToolBar { background-color: #404040; }")

        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.show_placeholder()

    def show_placeholder(self):
        """Show enhanced placeholder text"""
        self.canvas.clear_plot()
        self.canvas.axes.text(
            0.5,
            0.5,
            "Run Simulation to View\nAnalysis Results",
            transform=self.canvas.axes.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="#888",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#404040", alpha=0.8),
        )
        # Remove title to save vertical space
        self.canvas.draw()

    def update_results(self, results: Dict[str, np.ndarray]):
        """Update with new simulation results"""
        self.current_results = results
        self.plot_selector.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.show_grid.setEnabled(True)
        self.show_markers.setEnabled(True)
        self.update_plot()

    def update_plot(self):
        """Update the current plot with enhanced styling"""
        if not self.current_results:
            self.show_placeholder()
            return

        self.canvas.clear_plot()

        selected_plot = self.plot_selector.currentText()
        wavelengths = self.current_results["wavelengths"]

        # Log plot type change
        self.safe_log(
            f"Analysis plot changed to: {selected_plot} ({len(wavelengths)} wavelength points)"
        )

        # Plot styling options
        line_width = 2.5
        marker_style = "o" if self.show_markers.isChecked() else None
        marker_size = 4 if self.show_markers.isChecked() else 0
        alpha = 0.8

        if selected_plot == "Transmission Spectrum":
            self.canvas.axes.plot(
                wavelengths,
                self.current_results["through_dB"],
                "#4CAF50",
                linewidth=line_width,
                label="Through Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.plot(
                wavelengths,
                self.current_results["drop_dB"],
                "#F44336",
                linewidth=line_width,
                label="Drop Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.set_ylabel("Transmission (dB)", fontsize=12, color="white")
            # Remove title to save vertical space

        elif selected_plot == "Phase Response":
            self.canvas.axes.plot(
                wavelengths,
                self.current_results["through_phase"] / np.pi,
                "#2196F3",
                linewidth=line_width,
                label="Through Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.plot(
                wavelengths,
                self.current_results["drop_phase"] / np.pi,
                "#FF9800",
                linewidth=line_width,
                label="Drop Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.set_ylabel("Phase (π rad)", fontsize=12, color="white")
            # Remove title to save vertical space

        elif selected_plot == "Group Delay":
            wl_delay = wavelengths[:-1]
            self.canvas.axes.plot(
                wl_delay,
                self.current_results["through_delay"] * 1e12,
                "#9C27B0",
                linewidth=line_width,
                label="Through Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.plot(
                wl_delay,
                self.current_results["drop_delay"] * 1e12,
                "#00BCD4",
                linewidth=line_width,
                label="Drop Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.set_ylabel("Group Delay (ps)", fontsize=12, color="white")
            # Remove title to save vertical space

        elif selected_plot == "Dispersion":
            wl_disp = wavelengths[:-2]
            self.canvas.axes.plot(
                wl_disp,
                self.current_results["through_dispersion"] * 1e12,
                "#795548",
                linewidth=line_width,
                label="Through Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.plot(
                wl_disp,
                self.current_results["drop_dispersion"] * 1e12,
                "#607D8B",
                linewidth=line_width,
                label="Drop Port",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.set_ylabel("Dispersion (ps²/nm)", fontsize=12, color="white")
            # Remove title to save vertical space

        elif selected_plot == "Effective Index":
            self.canvas.axes.plot(
                wavelengths,
                self.current_results["neff"],
                "#FFC107",
                linewidth=line_width,
                label="Effective Index",
                marker=marker_style,
                markersize=marker_size,
                alpha=alpha,
            )
            self.canvas.axes.set_ylabel("Effective Index", fontsize=12, color="white")
            # Remove title to save vertical space

        # Enhanced plot styling
        self.canvas.axes.set_xlabel("Wavelength (nm)", fontsize=12, color="white")

        # Grid styling
        if self.show_grid.isChecked():
            self.canvas.axes.grid(True, alpha=0.3, color="gray", linestyle="--")

        # Enhanced legend
        legend = self.canvas.axes.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="#404040",
            edgecolor="white",
        )
        legend.get_frame().set_alpha(0.9)

        # Tight layout to maximize plot area and eliminate whitespace
        self.canvas.fig.tight_layout(rect=[0, 0, 1, 1], pad=0.1)
        self.canvas.draw()

        print(f"Enhanced analysis plot updated: {selected_plot}")

    def export_plot(self):
        """Export the current plot with high quality"""
        self.safe_log("Export Plot button clicked")

        if not self.current_results:
            self.safe_log("Export failed: No simulation data available")
            QMessageBox.warning(self, "No Data", "No simulation results to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            f"ring_analysis_{self.plot_selector.currentText().lower().replace(' ', '_')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
        )

        if filename:
            self.canvas.fig.savefig(
                filename, dpi=300, bbox_inches="tight", facecolor="#2e2e2e", edgecolor="white"
            )
            self.safe_log(f"Plot exported to: {filename}")
            QMessageBox.information(
                self, "Export Successful", f"High-quality plot saved to {filename}"
            )


class ControlPanel(QWidget):
    """Enhanced control panel with better styling"""

    parameters_changed = Signal()

    def __init__(self):
        super().__init__()
        self.log_callback = None
        self.setup_ui()

    def safe_log(self, message):
        """Safely log a message if callback is available"""
        if self.log_callback:
            self.log_callback(message)

    def log_parameter_change(self, parameter_name, value, unit=""):
        """Log parameter changes with clear descriptions"""
        self.safe_log(f"Parameter changed: {parameter_name} = {value}{unit}")

    def setup_ui(self):
        layout = QVBoxLayout()

        # Enhanced title
        title = QLabel("Ring Resonator Parameters")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            """
            QLabel {
                font-size: 16px; 
                font-weight: bold; 
                color: #2196F3; 
                margin: 10px;
                padding: 10px;
                background-color: #404040;
                border-radius: 5px;
            }
        """
        )
        layout.addWidget(title)

        # Scroll area with enhanced styling
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: #2e2e2e;
            }
            QScrollBar:vertical {
                background-color: #404040;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #2196F3;
                border-radius: 6px;
                min-height: 20px;
            }
        """
        )
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # Enhanced group box styling
        group_style = """
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 2px solid #555;
                border-radius: 8px;
                margin: 10px 0px;
                padding: 10px;
                background-color: #404040;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                color: #2196F3;
            }
        """

        # System Configuration
        sys_group = QGroupBox("System Configuration")
        sys_group.setStyleSheet(group_style)
        sys_layout = QGridLayout()

        sys_layout.addWidget(QLabel("Number of Rings:"), 0, 0)
        self.num_rings_spin = QSpinBox()
        self.num_rings_spin.setRange(1, 10)
        self.num_rings_spin.setValue(2)
        self.num_rings_spin.valueChanged.connect(self.update_ring_controls)
        self.num_rings_spin.setStyleSheet(
            "QSpinBox { background-color: #555; color: white; border: 1px solid #777; }"
        )
        sys_layout.addWidget(self.num_rings_spin, 0, 1)

        sys_group.setLayout(sys_layout)
        scroll_layout.addWidget(sys_group)

        # Ring Parameters
        self.ring_group = QGroupBox("Ring Parameters")
        self.ring_group.setStyleSheet(group_style)
        self.ring_layout = QGridLayout()
        self.ring_controls = []
        self.coupling_controls = []
        self.phase_controls = []

        self.update_ring_controls()
        self.ring_group.setLayout(self.ring_layout)
        scroll_layout.addWidget(self.ring_group)

        # Wavelength Configuration
        wl_group = QGroupBox("Wavelength Configuration")
        wl_group.setStyleSheet(group_style)
        wl_layout = QGridLayout()

        spinbox_style = (
            "QDoubleSpinBox { background-color: #555; color: white; border: 1px solid #777; }"
        )

        wl_layout.addWidget(QLabel("Start (nm):"), 0, 0)
        self.wl_start_spin = QDoubleSpinBox()
        self.wl_start_spin.setRange(1000, 2000)
        self.wl_start_spin.setValue(1520)
        self.wl_start_spin.valueChanged.connect(self.parameters_changed)
        self.wl_start_spin.valueChanged.connect(
            lambda v: self.log_parameter_change("Wavelength Start", v, " nm")
        )
        self.wl_start_spin.setStyleSheet(spinbox_style)
        wl_layout.addWidget(self.wl_start_spin, 0, 1)

        wl_layout.addWidget(QLabel("Stop (nm):"), 1, 0)
        self.wl_stop_spin = QDoubleSpinBox()
        self.wl_stop_spin.setRange(1000, 2000)
        self.wl_stop_spin.setValue(1570)
        self.wl_stop_spin.valueChanged.connect(self.parameters_changed)
        self.wl_stop_spin.valueChanged.connect(
            lambda v: self.log_parameter_change("Wavelength Stop", v, " nm")
        )
        self.wl_stop_spin.setStyleSheet(spinbox_style)
        wl_layout.addWidget(self.wl_stop_spin, 1, 1)

        wl_layout.addWidget(QLabel("Resolution (nm):"), 2, 0)
        self.wl_res_spin = QDoubleSpinBox()
        self.wl_res_spin.setRange(0.001, 1.0)
        self.wl_res_spin.setValue(0.01)
        self.wl_res_spin.setDecimals(3)
        self.wl_res_spin.valueChanged.connect(self.parameters_changed)
        self.wl_res_spin.valueChanged.connect(
            lambda v: self.log_parameter_change("Wavelength Resolution", v, " nm")
        )
        self.wl_res_spin.setStyleSheet(spinbox_style)
        wl_layout.addWidget(self.wl_res_spin, 2, 1)

        wl_group.setLayout(wl_layout)
        scroll_layout.addWidget(wl_group)

        # Material Properties
        mat_group = QGroupBox("Material Properties")
        mat_group.setStyleSheet(group_style)
        mat_layout = QGridLayout()

        mat_layout.addWidget(QLabel("Loss (dB/cm):"), 0, 0)
        self.loss_spin = QDoubleSpinBox()
        self.loss_spin.setRange(0.0, 100.0)
        self.loss_spin.setValue(4.0)
        self.loss_spin.valueChanged.connect(self.parameters_changed)
        self.loss_spin.valueChanged.connect(
            lambda v: self.log_parameter_change("Loss", v, " dB/cm")
        )
        self.loss_spin.setStyleSheet(spinbox_style)
        mat_layout.addWidget(self.loss_spin, 0, 1)

        mat_layout.addWidget(QLabel("Effective Index n₀:"), 1, 0)
        self.neff_0_spin = QDoubleSpinBox()
        self.neff_0_spin.setRange(1.0, 5.0)
        self.neff_0_spin.setValue(4.077)
        self.neff_0_spin.setDecimals(6)
        self.neff_0_spin.setSingleStep(0.001)
        self.neff_0_spin.valueChanged.connect(self.parameters_changed)
        self.neff_0_spin.valueChanged.connect(
            lambda v: self.log_parameter_change("Effective Index n₀", v)
        )
        self.neff_0_spin.setStyleSheet(spinbox_style)
        mat_layout.addWidget(self.neff_0_spin, 1, 1)

        mat_layout.addWidget(QLabel("Effective Index n₁:"), 2, 0)
        self.neff_1_spin = QDoubleSpinBox()
        self.neff_1_spin.setRange(-5.0, 5.0)
        self.neff_1_spin.setValue(-0.983)
        self.neff_1_spin.setDecimals(6)
        self.neff_1_spin.setSingleStep(0.001)
        self.neff_1_spin.valueChanged.connect(self.parameters_changed)
        self.neff_1_spin.valueChanged.connect(
            lambda v: self.log_parameter_change("Effective Index n₁", v)
        )
        self.neff_1_spin.setStyleSheet(spinbox_style)
        mat_layout.addWidget(self.neff_1_spin, 2, 1)

        mat_layout.addWidget(QLabel("Effective Index n₂:"), 3, 0)
        self.neff_2_spin = QDoubleSpinBox()
        self.neff_2_spin.setRange(-1.0, 1.0)
        self.neff_2_spin.setValue(-0.046)
        self.neff_2_spin.setDecimals(6)
        self.neff_2_spin.setSingleStep(0.001)
        self.neff_2_spin.valueChanged.connect(self.parameters_changed)
        self.neff_2_spin.valueChanged.connect(
            lambda v: self.log_parameter_change("Effective Index n₂", v)
        )
        self.neff_2_spin.setStyleSheet(spinbox_style)
        mat_layout.addWidget(self.neff_2_spin, 3, 1)

        mat_group.setLayout(mat_layout)
        scroll_layout.addWidget(mat_group)

        # Control Buttons
        btn_group = QGroupBox("Controls")
        btn_group.setStyleSheet(group_style)
        btn_layout = QVBoxLayout()

        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: none;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """
        )
        btn_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #404040;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 5px;
            }
        """
        )
        btn_layout.addWidget(self.progress_bar)

        btn_layout2 = QHBoxLayout()
        button_style = """
            QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid #777;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """

        self.save_btn = QPushButton("Save Config")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setStyleSheet(button_style)
        btn_layout2.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Config")
        self.load_btn.clicked.connect(self.load_config)
        self.load_btn.setStyleSheet(button_style)
        btn_layout2.addWidget(self.load_btn)

        btn_layout.addLayout(btn_layout2)
        btn_group.setLayout(btn_layout)
        scroll_layout.addWidget(btn_group)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self.setLayout(layout)

    def update_ring_controls(self):
        """Update ring parameter controls with enhanced styling"""
        # Clear existing controls
        for i in reversed(range(self.ring_layout.count())):
            item = self.ring_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        self.ring_controls.clear()
        self.coupling_controls.clear()
        self.phase_controls.clear()

        num_rings = self.num_rings_spin.value()

        # Log ring configuration change
        self.safe_log(f"Ring configuration changed: {num_rings} rings")

        row = 0

        spinbox_style = (
            "QDoubleSpinBox { background-color: #555; color: white; border: 1px solid #777; }"
        )

        # Ring radii with vernier design (3/4 multipliers)
        for i in range(num_rings):
            label = QLabel(f"Ring {i+1} Radius (μm):")
            label.setStyleSheet("color: white;")
            self.ring_layout.addWidget(label, row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(1.0, 200.0)
            # Set vernier ring radii: R1=35, R2=R1*3/4, R3=R2*3/4, etc.
            base_radius = 35.0
            radius_value = base_radius * (0.75**i)
            spin.setValue(radius_value)
            spin.valueChanged.connect(self.parameters_changed)
            spin.valueChanged.connect(
                lambda v, idx=i: self.log_parameter_change(f"Ring {idx+1} Radius", v, " μm")
            )
            spin.setStyleSheet(spinbox_style)
            self.ring_controls.append(spin)
            self.ring_layout.addWidget(spin, row, 1)
            row += 1

        # Coupling coefficients
        for i in range(num_rings + 1):
            if i == 0:
                label_text = "Input Coupling κ₁:"
            elif i == num_rings:
                label_text = f"Output Coupling κ{i+1}:"
            else:
                label_text = f"Ring {i}-{i+1} Coupling κ{i+1}:"

            label = QLabel(label_text)
            label.setStyleSheet("color: white;")
            self.ring_layout.addWidget(label, row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1.0)
            spin.setValue(0.1 if i == 0 or i == num_rings else 0.001)
            spin.setDecimals(3)
            spin.valueChanged.connect(self.parameters_changed)
            coupling_name = f"κ{i+1}" if i == 0 or i == num_rings else f"κ{i+1} (Ring {i}-{i+1})"
            spin.valueChanged.connect(
                lambda v, name=coupling_name: self.log_parameter_change(name, v)
            )
            spin.setStyleSheet(spinbox_style)
            self.coupling_controls.append(spin)
            self.ring_layout.addWidget(spin, row, 1)
            row += 1

        # Phase shifts
        for i in range(num_rings):
            label = QLabel(f"Ring {i+1} Phase (π rad):")
            label.setStyleSheet("color: white;")
            self.ring_layout.addWidget(label, row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-1.0, 1.0)
            spin.setValue(0.0)
            spin.setDecimals(3)
            spin.valueChanged.connect(self.parameters_changed)
            spin.valueChanged.connect(
                lambda v, idx=i: self.log_parameter_change(f"Ring {idx+1} Phase", v, " π rad")
            )
            spin.setStyleSheet(spinbox_style)
            self.phase_controls.append(spin)
            self.ring_layout.addWidget(spin, row, 1)
            row += 1

        self.parameters_changed.emit()

    def get_configuration(self) -> Dict:
        """Get current configuration"""
        return {
            "ring_radii_um": [spin.value() for spin in self.ring_controls],
            "coupling_coeffs": [spin.value() for spin in self.coupling_controls],
            "phase_shifts_rad": [spin.value() * np.pi for spin in self.phase_controls],
            "wavelength_start_nm": self.wl_start_spin.value(),
            "wavelength_stop_nm": self.wl_stop_spin.value(),
            "wavelength_resolution_nm": self.wl_res_spin.value(),
            "loss_db_per_cm": self.loss_spin.value(),
            "neff_coeffs": [
                self.neff_0_spin.value(),
                self.neff_1_spin.value(),
                self.neff_2_spin.value(),
            ],
        }

    def save_config(self):
        """Save configuration"""
        self.safe_log("Save Configuration button clicked")

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "ring_config.json", "JSON Files (*.json)"
        )

        if filename:
            config = self.get_configuration()
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
            self.safe_log(f"Configuration saved to: {filename}")
            QMessageBox.information(self, "Success", f"Configuration saved to {filename}")

    def load_config(self):
        """Load configuration"""
        self.safe_log("Load Configuration button clicked")

        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )

        if filename:
            with open(filename, "r") as f:
                config = json.load(f)
            # Set values from config
            self.num_rings_spin.setValue(len(config.get("ring_radii_um", [])))
            # Update other controls...
            self.safe_log(f"Configuration loaded from: {filename}")
            QMessageBox.information(self, "Success", f"Configuration loaded from {filename}")


class SimulationWorker(QObject):
    """Worker for running simulations"""

    progress_updated = Signal(int)
    log_message = Signal(str)
    simulation_finished = Signal(dict)
    simulation_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.system = RingResonatorSystem()
        self.analyzer = RingAnalyzer()

    def run_simulation(self, config: Dict):
        """Run the simulation"""
        try:
            self.log_message.emit("Starting simulation...")
            self.progress_updated.emit(10)

            # Configure system
            self.system.configure(**config)
            self.progress_updated.emit(30)

            # Run analysis
            self.log_message.emit("Running system analysis...")
            wavelengths, t1N, r1N, rN1, tN1 = self.system.analyze_system()
            self.progress_updated.emit(60)

            # Calculate results
            self.log_message.emit("Calculating power response...")
            drop_dB, through_dB, total_dB = self.analyzer.calculate_power_response(t1N, r1N)
            drop_phase, through_phase = self.analyzer.calculate_phase_response(t1N, r1N)
            self.progress_updated.emit(80)

            # Calculate group delay and dispersion
            self.log_message.emit("Calculating group delay and dispersion...")
            c = 299792458
            wavelengths_m = wavelengths * 1e-9
            omega = 2 * np.pi * c / wavelengths_m
            through_delay, drop_delay = self.analyzer.calculate_group_delay(
                wavelengths_m, through_phase, drop_phase
            )
            through_dispersion, drop_dispersion = self.analyzer.calculate_dispersion(
                (through_delay, drop_delay), omega
            )
            self.progress_updated.emit(90)

            # Calculate effective index
            neff = (
                self.system.neff_coeffs[0]
                + self.system.neff_coeffs[1] * (wavelengths_m * 1e6)
                + self.system.neff_coeffs[2] * (wavelengths_m * 1e6) ** 2
            )
            self.progress_updated.emit(100)

            # Prepare results
            results = {
                "wavelengths": wavelengths,
                "drop_dB": drop_dB,
                "through_dB": through_dB,
                "drop_phase": drop_phase,
                "through_phase": through_phase,
                "through_delay": through_delay,
                "drop_delay": drop_delay,
                "through_dispersion": through_dispersion,
                "drop_dispersion": drop_dispersion,
                "neff": neff,
            }

            self.log_message.emit("Simulation completed successfully!")
            self.simulation_finished.emit(results)

        except Exception as e:
            self.simulation_error.emit(str(e))
            self.log_message.emit(f"Simulation failed: {str(e)}")


class RingResonatorStudio(QMainWindow):
    """Enhanced main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ring Resonator Design Studio - Enhanced")
        self.setMinimumSize(1400, 900)

        # Apply dark theme to main window
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #2e2e2e;
                color: white;
            }
            QMenuBar {
                background-color: #404040;
                color: white;
                border-bottom: 1px solid #555;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #2196F3;
            }
            QStatusBar {
                background-color: #404040;
                color: white;
                border-top: 1px solid #555;
            }
            QLabel {
                color: white;
            }
        """
        )

        # Setup worker thread
        self.worker_thread = QThread()
        self.worker = SimulationWorker()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.log_message.connect(self.add_log_message)
        self.worker.simulation_finished.connect(self.simulation_finished)
        self.worker.simulation_error.connect(self.simulation_error)

        self.worker_thread.start()

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Setup the enhanced UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout using splitter for better control
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Controls (reduced width to give more space to plots)
        self.control_panel = ControlPanel()
        self.control_panel.setMaximumWidth(320)
        self.control_panel.setMinimumWidth(300)
        main_splitter.addWidget(self.control_panel)

        # Right panel - Plots
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Geometry plot
        self.geometry_plot = InteractiveGeometryPlot()
        right_splitter.addWidget(self.geometry_plot)

        # Analysis plot
        self.analysis_plot = InteractiveAnalysisPlot()
        right_splitter.addWidget(self.analysis_plot)

        # Set initial splitter sizes to maximize plot area
        right_splitter.setSizes([400, 400])

        main_splitter.addWidget(right_splitter)
        # Give more space to plots by reducing control panel width
        main_splitter.setSizes([320, 1200])

        # Add log area at bottom
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout()

        log_group = QGroupBox("Simulation Log")
        log_group.setStyleSheet(
            """
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin: 5px;
                padding: 5px;
                background-color: #404040;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                color: #2196F3;
            }
        """
        )
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(80)  # Reduced height to give more space to plots
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #2e2e2e;
                color: #4CAF50;
                border: 1px solid #555;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """
        )
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        bottom_layout.addWidget(log_group)

        bottom_panel.setLayout(bottom_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        main_layout.addWidget(bottom_panel)

        central_widget.setLayout(main_layout)

        # Status bar
        self.statusBar().showMessage("Ready - Enhanced Ring Resonator Design Studio")

    def connect_signals(self):
        """Connect signals"""
        self.control_panel.parameters_changed.connect(self.update_geometry)
        self.control_panel.run_btn.clicked.connect(self.run_simulation)

        # Set up logging callbacks
        self.control_panel.log_callback = self.add_log_message
        self.geometry_plot.log_callback = self.add_log_message
        self.analysis_plot.log_callback = self.add_log_message

        # Initial geometry update
        self.update_geometry()

    def update_geometry(self):
        """Update geometry plot"""
        config = self.control_panel.get_configuration()
        self.geometry_plot.update_geometry(
            config["ring_radii_um"], config["coupling_coeffs"], config["phase_shifts_rad"]
        )

    def run_simulation(self):
        """Run simulation"""
        config = self.control_panel.get_configuration()

        # Log simulation initiation with parameters
        self.add_log_message("Run Simulation button clicked")
        self.add_log_message(
            f"Simulation parameters: {len(config['ring_radii_um'])} rings, "
            f"λ={config['wavelength_start_nm']}-{config['wavelength_stop_nm']} nm, "
            f"Δλ={config['wavelength_resolution_nm']} nm, "
            f"Loss={config['loss_db_per_cm']} dB/cm"
        )

        self.control_panel.run_btn.setEnabled(False)
        self.control_panel.progress_bar.setVisible(True)
        self.add_log_message("Starting simulation...")

        # Run simulation in worker thread
        QTimer.singleShot(0, lambda: self.worker.run_simulation(config))

    def update_progress(self, value: int):
        """Update progress bar"""
        self.control_panel.progress_bar.setValue(value)

    def add_log_message(self, message: str):
        """Add log message"""
        self.log_text.append(message)

    def simulation_finished(self, results: Dict):
        """Handle simulation completion"""
        self.control_panel.run_btn.setEnabled(True)
        self.control_panel.progress_bar.setVisible(False)
        self.statusBar().showMessage("Simulation completed successfully")

        # Update analysis plot
        self.analysis_plot.update_results(results)

        self.add_log_message("Simulation completed successfully!")
        print(f"Enhanced simulation completed with {len(results['wavelengths'])} wavelength points")

    def simulation_error(self, error: str):
        """Handle simulation error"""
        self.control_panel.run_btn.setEnabled(True)
        self.control_panel.progress_bar.setVisible(False)
        self.statusBar().showMessage("Simulation failed")

        QMessageBox.critical(self, "Simulation Error", f"Simulation failed: {error}")

    def closeEvent(self, event):
        """Handle close event"""
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Ring Resonator Design Studio - Enhanced")

    # Set dark theme for entire application
    app.setStyleSheet(
        """
        QApplication {
            background-color: #2e2e2e;
            color: white;
        }
        QWidget {
            background-color: #2e2e2e;
            color: white;
        }
        QCheckBox {
            color: white;
        }
        QCheckBox::indicator {
            background-color: #555;
            border: 1px solid #777;
        }
        QCheckBox::indicator:checked {
            background-color: #2196F3;
        }
    """
    )

    window = RingResonatorStudio()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

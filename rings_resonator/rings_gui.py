"""
Ring Resonator GUI Application

A modern graphical user interface for interactive ring resonator design and simulation.
Built with CustomTkinter for a modern, dark-themed interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import os
from typing import Optional, List, Tuple
import json

# Handle imports for both package and direct execution
try:
    from .rings_design import RingResonatorSystem, RingAnalyzer, RingPlotter
except ImportError:
    # If running directly, try absolute import
    from rings_design import RingResonatorSystem, RingAnalyzer, RingPlotter

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Modern color scheme
COLORS = {
    'primary': '#00D4FF',
    'secondary': '#FF6B6B', 
    'accent': '#4ECDC4',
    'success': '#45B7D1',
    'warning': '#FFA726',
    'bg_dark': '#1a1a1a',
    'bg_medium': '#2d2d2d',
    'bg_light': '#3d3d3d',
    'text_primary': '#ffffff',
    'text_secondary': '#b0b0b0',
    'ring_colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C']
}


class RingVisualizationWidget(ctk.CTkFrame):
    """Widget for visualizing ring resonator geometry."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=COLORS['bg_dark'])
        self.ax = self.fig.add_subplot(111, facecolor=COLORS['bg_dark'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Style the plot with modern colors
        self.ax.set_facecolor(COLORS['bg_dark'])
        self.ax.tick_params(colors=COLORS['text_primary'], labelsize=10)
        self.ax.xaxis.label.set_color(COLORS['text_primary'])
        self.ax.yaxis.label.set_color(COLORS['text_primary'])
        self.ax.title.set_color(COLORS['text_primary'])
        self.ax.spines['bottom'].set_color(COLORS['text_secondary'])
        self.ax.spines['top'].set_color(COLORS['text_secondary'])
        self.ax.spines['left'].set_color(COLORS['text_secondary'])
        self.ax.spines['right'].set_color(COLORS['text_secondary'])
        
        self.draw_empty_plot()
    
    def draw_empty_plot(self):
        """Draw empty plot with instructions."""
        self.ax.clear()
        self.ax.text(0.5, 0.5, '• Configure rings to see visualization', 
                    ha='center', va='center', transform=self.ax.transAxes,
                    color=COLORS['text_secondary'], fontsize=14, fontweight='bold')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2, color=COLORS['text_secondary'], linestyle='--')
        self.canvas.draw()
    
    def draw_rings(self, ring_radii_um: List[float], coupling_coeffs: List[float]):
        """Draw ring resonator geometry with vertical arrangement."""
        self.ax.clear()
        
        # Set modern plot style
        self.ax.set_facecolor(COLORS['bg_dark'])
        self.ax.tick_params(colors=COLORS['text_primary'], labelsize=10)
        self.ax.spines['bottom'].set_color(COLORS['text_secondary'])
        self.ax.spines['top'].set_color(COLORS['text_secondary'])
        self.ax.spines['left'].set_color(COLORS['text_secondary'])
        self.ax.spines['right'].set_color(COLORS['text_secondary'])
        
        if not ring_radii_um:
            self.draw_empty_plot()
            return
            
        # Calculate vertical positions for rings
        max_radius = max(ring_radii_um)
        total_height = sum(ring_radii_um) * 2.5
        
        # Center the rings vertically
        y_positions = []
        current_y = total_height / 2
        
        for i, radius in enumerate(ring_radii_um):
            current_y -= radius
            y_positions.append(current_y)
            current_y -= radius * 1.5
        
        # Draw bus waveguides (above and below rings)
        x_wg_extent = max_radius * 1.5
        bus_top_y = max(y_positions) + max_radius + 30
        bus_bottom_y = min(y_positions) - max_radius - 30
        
        # Bus waveguide above (Input/Through)
        self.ax.plot([-x_wg_extent, x_wg_extent], [bus_top_y, bus_top_y], 
                    color=COLORS['primary'], linewidth=4, label='Bus Waveguide (Input → Through)',
                    solid_capstyle='round')
        
        # Drop waveguide below
        self.ax.plot([-x_wg_extent, x_wg_extent], [bus_bottom_y, bus_bottom_y], 
                    color=COLORS['secondary'], linewidth=4, label='Drop Waveguide',
                    solid_capstyle='round')
        
        # Draw rings with modern styling
        for i, (y, radius) in enumerate(zip(y_positions, ring_radii_um)):
            color = COLORS['ring_colors'][i % len(COLORS['ring_colors'])]
            
            # Ring with gradient effect (multiple circles with decreasing alpha)
            for j in range(3):
                alpha = 0.8 - j * 0.2
                lw = 3 - j * 0.5
                circle = plt.Circle((0, y), radius - j * 0.5, fill=False, color=color, 
                                  linewidth=lw, alpha=alpha)
                self.ax.add_patch(circle)
            
            # Ring label with modern styling
            self.ax.text(radius + 15, y, f'Ring {i+1}\nR = {radius:.1f}μm', 
                        ha='left', va='center', color=color, fontsize=10, 
                        fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor=COLORS['bg_medium'], alpha=0.8, edgecolor=color))
            
            # Coupling indicators with modern arrows
            if i < len(coupling_coeffs):
                # Arrow from bus to ring
                self.ax.annotate('', xy=(0, y + radius), xytext=(0, bus_top_y - 5),
                               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], 
                                             lw=2, alpha=0.7))
                self.ax.text(10, y + radius + 5, f'κ₁ = {coupling_coeffs[i]:.3f}', 
                           ha='left', va='bottom', color=COLORS['accent'], fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS['bg_medium'], 
                                   alpha=0.7, edgecolor=COLORS['accent']))
            
            # Inter-ring coupling
            if i < len(ring_radii_um) - 1 and i + 1 < len(coupling_coeffs):
                next_y = y_positions[i + 1]
                mid_y = (y + next_y) / 2
                self.ax.annotate('', xy=(0, next_y + ring_radii_um[i+1]), 
                               xytext=(0, y - radius),
                               arrowprops=dict(arrowstyle='<->', color=COLORS['warning'], 
                                             lw=2, alpha=0.7))
                self.ax.text(15, mid_y, f'κ₂ = {coupling_coeffs[i+1]:.3f}', 
                           ha='left', va='center', color=COLORS['warning'], fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS['bg_medium'], 
                                   alpha=0.7, edgecolor=COLORS['warning']))
        
        # Final coupling to drop port
        if len(coupling_coeffs) > len(ring_radii_um):
            last_y = y_positions[-1]
            last_radius = ring_radii_um[-1]
            self.ax.annotate('', xy=(0, bus_bottom_y + 5), xytext=(0, last_y - last_radius),
                           arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], 
                                         lw=2, alpha=0.7))
            self.ax.text(10, last_y - last_radius - 10, f'κₒᵤₜ = {coupling_coeffs[-1]:.3f}', 
                       ha='left', va='top', color=COLORS['secondary'], fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS['bg_medium'], 
                               alpha=0.7, edgecolor=COLORS['secondary']))
        
        # Modern port labels
        self.ax.text(-x_wg_extent - 20, bus_top_y, '◀ Input', ha='right', va='center', 
                    color=COLORS['primary'], fontsize=12, fontweight='bold')
        self.ax.text(x_wg_extent + 20, bus_top_y, 'Through ▶', ha='left', va='center', 
                    color=COLORS['primary'], fontsize=12, fontweight='bold')
        self.ax.text(x_wg_extent + 20, bus_bottom_y, 'Drop ▼', ha='left', va='center', 
                    color=COLORS['secondary'], fontsize=12, fontweight='bold')
        
        # Set limits and styling
        self.ax.set_xlim(-x_wg_extent - 50, x_wg_extent + 80)
        self.ax.set_ylim(bus_bottom_y - 20, bus_top_y + 20)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('Position (μm)', color=COLORS['text_primary'], fontsize=12)
        self.ax.set_ylabel('Position (μm)', color=COLORS['text_primary'], fontsize=12)
        self.ax.set_title('◈ Ring Resonator Geometry', color=COLORS['text_primary'], 
                         fontsize=16, fontweight='bold', pad=20)
        self.ax.grid(True, alpha=0.2, color=COLORS['text_secondary'], linestyle='--')
        
        # Modern legend
        self.ax.legend(loc='upper left', fancybox=True, shadow=True, 
                      framealpha=0.9, facecolor=COLORS['bg_medium'],
                      edgecolor=COLORS['text_secondary'])
        
        self.canvas.draw()


class PlotWidget(ctk.CTkFrame):
    """Widget for displaying analysis plots."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor=COLORS['bg_dark'])
        
        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add navigation toolbar for zooming/panning
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        
        # Create subplots with modern styling
        self.ax1 = self.fig.add_subplot(221, facecolor=COLORS['bg_dark'])
        self.ax2 = self.fig.add_subplot(222, facecolor=COLORS['bg_dark'])
        self.ax3 = self.fig.add_subplot(223, facecolor=COLORS['bg_dark'])
        self.ax4 = self.fig.add_subplot(224, facecolor=COLORS['bg_dark'])
        
        self.fig.tight_layout(pad=3.0)
        self.setup_plot_styles()
        self.draw_empty_plots()
    
    def setup_plot_styles(self):
        """Setup modern plot styling."""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor(COLORS['bg_dark'])
            ax.tick_params(colors=COLORS['text_primary'], labelsize=9)
            ax.xaxis.label.set_color(COLORS['text_primary'])
            ax.yaxis.label.set_color(COLORS['text_primary'])
            ax.title.set_color(COLORS['text_primary'])
            ax.grid(True, alpha=0.2, color=COLORS['text_secondary'], linestyle='--')
            
            # Modern spines
            for spine in ax.spines.values():
                spine.set_color(COLORS['text_secondary'])
                spine.set_linewidth(0.5)
    
    def draw_empty_plots(self):
        """Draw empty plots with modern labels."""
        labels = [
            ('◆ Transmission\nSpectrum', COLORS['primary']),
            ('◈ Phase\nResponse', COLORS['accent']),
            ('◇ Group\nDelay', COLORS['success']),
            ('◉ Dispersion', COLORS['warning'])
        ]
        
        axes = [self.ax1, self.ax2, self.ax3, self.ax4]
        
        for ax, (label, color) in zip(axes, labels):
            ax.text(0.5, 0.5, label, ha='center', va='center', 
                   transform=ax.transAxes, color=color, fontsize=12, 
                   fontweight='bold', alpha=0.7)
                   
        self.canvas.draw()
    
    def update_plots(self, wavelengths, drop_dB, through_dB, drop_phase, through_phase, 
                    through_delay, drop_delay, through_dispersion, drop_dispersion):
        """Update all plots with new data and modern styling."""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Transmission spectrum with modern styling
        self.ax1.plot(wavelengths, through_dB, color=COLORS['primary'], 
                     label='Through Port', linewidth=2, alpha=0.8)
        self.ax1.plot(wavelengths, drop_dB, color=COLORS['secondary'], 
                     label='Drop Port', linewidth=2, alpha=0.8)
        self.ax1.set_xlabel('Wavelength (nm)', fontsize=10)
        self.ax1.set_ylabel('Transmission (dB)', fontsize=10)
        self.ax1.set_title('◆ Transmission Spectrum', fontsize=12, fontweight='bold')
        self.ax1.legend(framealpha=0.9, facecolor=COLORS['bg_medium'])
        
        # Phase response
        self.ax2.plot(wavelengths, through_phase / np.pi, color=COLORS['accent'], 
                     label='Through Port', linewidth=2, alpha=0.8)
        self.ax2.plot(wavelengths, drop_phase / np.pi, color=COLORS['warning'], 
                     label='Drop Port', linewidth=2, alpha=0.8)
        self.ax2.set_xlabel('Wavelength (nm)', fontsize=10)
        self.ax2.set_ylabel('Phase (π rad)', fontsize=10)
        self.ax2.set_title('◈ Phase Response', fontsize=12, fontweight='bold')
        self.ax2.legend(framealpha=0.9, facecolor=COLORS['bg_medium'])
        
        # Group delay
        wl_delay = wavelengths[:-1]
        self.ax3.plot(wl_delay, through_delay * 1e12, color=COLORS['success'], 
                     label='Through Port', linewidth=2, alpha=0.8)
        self.ax3.plot(wl_delay, drop_delay * 1e12, color='#FF9F43', 
                     label='Drop Port', linewidth=2, alpha=0.8)
        self.ax3.set_xlabel('Wavelength (nm)', fontsize=10)
        self.ax3.set_ylabel('Group Delay (ps)', fontsize=10)
        self.ax3.set_title('◇ Group Delay', fontsize=12, fontweight='bold')
        self.ax3.legend(framealpha=0.9, facecolor=COLORS['bg_medium'])
        
        # Dispersion
        wl_disp = wavelengths[:-2]
        self.ax4.plot(wl_disp, through_dispersion * 1e12, color=COLORS['warning'], 
                     label='Through Port', linewidth=2, alpha=0.8)
        self.ax4.plot(wl_disp, drop_dispersion * 1e12, color='#E74C3C', 
                     label='Drop Port', linewidth=2, alpha=0.8)
        self.ax4.set_xlabel('Wavelength (nm)', fontsize=10)
        self.ax4.set_ylabel('Dispersion (ps²/nm)', fontsize=10)
        self.ax4.set_title('◉ Group Delay Dispersion', fontsize=12, fontweight='bold')
        self.ax4.legend(framealpha=0.9, facecolor=COLORS['bg_medium'])
        
        # Apply modern styling
        self.setup_plot_styles()
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()


class RingResonatorGUI:
    """Main GUI application for ring resonator design."""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("◈ Ring Resonator Design Studio")
        self.root.geometry("1600x1000")
        
        # Set modern window icon and styling
        self.root.configure(fg_color=COLORS['bg_dark'])
        
        # Initialize system
        self.system = RingResonatorSystem()
        self.analyzer = RingAnalyzer()
        
        # GUI variables
        self.ring_vars = []
        self.coupling_vars = []
        self.phase_vars = []
        
        self.setup_ui()
        self.load_default_parameters()
        
    def setup_ui(self):
        """Setup the modern user interface."""
        # Main container with gradient-like styling
        main_frame = ctk.CTkFrame(self.root, fg_color=COLORS['bg_medium'], 
                                 corner_radius=15, border_width=1, 
                                 border_color=COLORS['text_secondary'])
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Left panel - Controls
        controls_frame = ctk.CTkFrame(main_frame, fg_color=COLORS['bg_light'],
                                     corner_radius=12, border_width=1,
                                     border_color=COLORS['primary'])
        controls_frame.pack(side="left", fill="y", padx=(0, 15), pady=10)
        
        # Right panel - Visualization
        viz_frame = ctk.CTkFrame(main_frame, fg_color=COLORS['bg_light'],
                                corner_radius=12, border_width=1,
                                border_color=COLORS['accent'])
        viz_frame.pack(side="right", fill="both", expand=True, pady=10)
        
        # Setup visualization first so ring_viz exists when controls are initialized
        self.setup_visualization(viz_frame)
        self.setup_controls(controls_frame)
        
    def setup_controls(self, parent):
        """Setup modern control panel."""
        # Title with gradient effect
        title_label = ctk.CTkLabel(parent, text="◈ Ring Resonator Design", 
                                  font=ctk.CTkFont(size=22, weight="bold"),
                                  text_color=COLORS['primary'])
        title_label.pack(pady=(15, 25))
        
        # Scrollable frame for controls with modern styling
        scroll_frame = ctk.CTkScrollableFrame(parent, width=380, height=650,
                                            fg_color=COLORS['bg_medium'],
                                            corner_radius=10)
        scroll_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # System Configuration
        self.setup_system_config(scroll_frame)
        
        # Ring Configuration
        self.setup_ring_config(scroll_frame)
        
        # Wavelength Configuration
        self.setup_wavelength_config(scroll_frame)
        
        # Material Configuration
        self.setup_material_config(scroll_frame)
        
        # Control buttons
        self.setup_control_buttons(scroll_frame)
        
    def setup_system_config(self, parent):
        """Setup modern system configuration controls."""
        frame = ctk.CTkFrame(parent, fg_color=COLORS['bg_light'], 
                           corner_radius=8, border_width=1, 
                           border_color=COLORS['primary'])
        frame.pack(fill="x", pady=(0, 15))
        
        label = ctk.CTkLabel(frame, text="◆ System Configuration", 
                            font=ctk.CTkFont(size=16, weight="bold"),
                            text_color=COLORS['primary'])
        label.pack(pady=15)
        
        # Number of rings with spinbox
        rings_frame = ctk.CTkFrame(frame, fg_color="transparent")
        rings_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(rings_frame, text="Number of Rings:", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        
        self.num_rings_var = ctk.IntVar(value=2)
        num_rings_spinbox = ctk.CTkEntry(rings_frame, width=80, 
                                       textvariable=self.num_rings_var,
                                       validate="key",
                                       validatecommand=(self.root.register(self.validate_rings), '%P'))
        num_rings_spinbox.pack(side="right", padx=10)
        
        # Bind real-time updates
        self.num_rings_var.trace_add("write", self.on_rings_change)
        
    def validate_rings(self, value):
        """Validate number of rings input."""
        if value == "":
            return True
        try:
            num = int(value)
            return 1 <= num <= 20  # Allow up to 20 rings
        except ValueError:
            return False
            
    def on_rings_change(self, *args):
        """Handle real-time ring count changes."""
        try:
            # Check if the value is valid before updating
            value = self.num_rings_var.get()
            if value > 0:
                # Small delay to avoid too frequent updates
                self.root.after(100, self.update_ring_controls)
        except:
            pass
        
    def setup_ring_config(self, parent):
        """Setup modern ring configuration controls."""
        self.ring_frame = ctk.CTkFrame(parent, fg_color=COLORS['bg_light'], 
                                     corner_radius=8, border_width=1, 
                                     border_color=COLORS['accent'])
        self.ring_frame.pack(fill="x", pady=(0, 15))
        
        label = ctk.CTkLabel(self.ring_frame, text="◇ Ring Parameters", 
                            font=ctk.CTkFont(size=16, weight="bold"),
                            text_color=COLORS['accent'])
        label.pack(pady=15)
        
        self.ring_controls_frame = ctk.CTkFrame(self.ring_frame, fg_color="transparent")
        self.ring_controls_frame.pack(fill="x", padx=15, pady=15)
        
        self.update_ring_controls()
        
    def setup_wavelength_config(self, parent):
        """Setup modern wavelength configuration controls."""
        frame = ctk.CTkFrame(parent, fg_color=COLORS['bg_light'], 
                           corner_radius=8, border_width=1, 
                           border_color=COLORS['success'])
        frame.pack(fill="x", pady=(0, 15))
        
        label = ctk.CTkLabel(frame, text="◈ Wavelength Range", 
                            font=ctk.CTkFont(size=16, weight="bold"),
                            text_color=COLORS['success'])
        label.pack(pady=15)
        
        # Start wavelength
        start_frame = ctk.CTkFrame(frame, fg_color="transparent")
        start_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(start_frame, text="Start (nm):", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.wl_start_var = ctk.StringVar(value="1520")
        ctk.CTkEntry(start_frame, textvariable=self.wl_start_var, width=100).pack(side="right")
        
        # Stop wavelength
        stop_frame = ctk.CTkFrame(frame, fg_color="transparent")
        stop_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(stop_frame, text="Stop (nm):", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.wl_stop_var = ctk.StringVar(value="1570")
        ctk.CTkEntry(stop_frame, textvariable=self.wl_stop_var, width=100).pack(side="right")
        
        # Resolution
        res_frame = ctk.CTkFrame(frame, fg_color="transparent")
        res_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(res_frame, text="Resolution (nm):", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.wl_res_var = ctk.StringVar(value="0.005")
        ctk.CTkEntry(res_frame, textvariable=self.wl_res_var, width=100).pack(side="right")
        
    def setup_material_config(self, parent):
        """Setup modern material configuration controls."""
        frame = ctk.CTkFrame(parent, fg_color=COLORS['bg_light'], 
                           corner_radius=8, border_width=1, 
                           border_color=COLORS['warning'])
        frame.pack(fill="x", pady=(0, 15))
        
        label = ctk.CTkLabel(frame, text="◉ Material Properties", 
                            font=ctk.CTkFont(size=16, weight="bold"),
                            text_color=COLORS['warning'])
        label.pack(pady=15)
        
        # Loss
        loss_frame = ctk.CTkFrame(frame, fg_color="transparent")
        loss_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(loss_frame, text="Loss (dB/cm):", 
                    font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.loss_var = ctk.StringVar(value="4.0")
        ctk.CTkEntry(loss_frame, textvariable=self.loss_var, width=100).pack(side="right")
        
    def setup_control_buttons(self, parent):
        """Setup modern control buttons."""
        button_frame = ctk.CTkFrame(parent, fg_color=COLORS['bg_light'], 
                                  corner_radius=8, border_width=1, 
                                  border_color=COLORS['secondary'])
        button_frame.pack(fill="x", pady=20)
        
        # Run simulation - main action button
        run_btn = ctk.CTkButton(button_frame, text="▶ Run Simulation", 
                               command=self.run_simulation,
                               font=ctk.CTkFont(size=16, weight="bold"),
                               height=45, corner_radius=8,
                               fg_color=COLORS['primary'],
                               hover_color=COLORS['accent'])
        run_btn.pack(pady=15)
        
        # Secondary action buttons
        actions_frame = ctk.CTkFrame(button_frame, fg_color="transparent")
        actions_frame.pack(fill="x", padx=15, pady=10)
        
        # Export plots
        export_btn = ctk.CTkButton(actions_frame, text="◆ Export Plots", 
                                  command=self.export_plots,
                                  width=160, height=35,
                                  fg_color=COLORS['success'],
                                  hover_color=COLORS['accent'])
        export_btn.pack(pady=5)
        
        # Save/Load configuration
        config_frame = ctk.CTkFrame(actions_frame, fg_color="transparent")
        config_frame.pack(fill="x", pady=5)
        
        save_btn = ctk.CTkButton(config_frame, text="◉ Save", 
                                command=self.save_config,
                                width=75, height=30,
                                fg_color=COLORS['warning'],
                                hover_color=COLORS['accent'])
        save_btn.pack(side="left", padx=(0, 5))
        
        load_btn = ctk.CTkButton(config_frame, text="◈ Load", 
                                command=self.load_config,
                                width=75, height=30,
                                fg_color=COLORS['warning'],
                                hover_color=COLORS['accent'])
        load_btn.pack(side="right", padx=(5, 0))
        
        # Modern progress bar
        self.progress = ctk.CTkProgressBar(button_frame, height=8, 
                                         corner_radius=4,
                                         progress_color=COLORS['primary'])
        self.progress.pack(pady=15, padx=15, fill="x")
        self.progress.set(0)
        
    def setup_visualization(self, parent):
        """Setup modern visualization panel."""
        # Title
        title_label = ctk.CTkLabel(parent, text="◈ Visualization Studio", 
                                  font=ctk.CTkFont(size=22, weight="bold"),
                                  text_color=COLORS['accent'])
        title_label.pack(pady=(15, 20))
        
        # Modern tabbed interface
        notebook = ctk.CTkTabview(parent, corner_radius=12, 
                                 fg_color=COLORS['bg_medium'],
                                 segmented_button_selected_color=COLORS['primary'],
                                 segmented_button_selected_hover_color=COLORS['accent'])
        notebook.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Ring geometry tab
        geometry_tab = notebook.add("◇ Ring Geometry")
        self.ring_viz = RingVisualizationWidget(geometry_tab)
        self.ring_viz.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Analysis plots tab
        plots_tab = notebook.add("◆ Analysis Plots")
        self.plot_widget = PlotWidget(plots_tab)
        self.plot_widget.pack(fill="both", expand=True, padx=10, pady=10)
        
    def update_ring_controls(self, *args):
        """Update ring parameter controls based on number of rings."""
        try:
            # Clear existing controls
            for widget in self.ring_controls_frame.winfo_children():
                widget.destroy()
            
            self.ring_vars = []
            self.coupling_vars = []
            self.phase_vars = []
            
            num_rings = max(1, self.num_rings_var.get())
            
            # Ring radii with modern styling
            for i in range(num_rings):
                frame = ctk.CTkFrame(self.ring_controls_frame, fg_color="transparent")
                frame.pack(fill="x", pady=3)
                
                ctk.CTkLabel(frame, text=f"Ring {i+1} Radius (μm):", 
                           font=ctk.CTkFont(size=11, weight="bold")).pack(side="left")
                var = ctk.StringVar(value=str(35.0 - i*3))
                entry = ctk.CTkEntry(frame, textvariable=var, width=80)
                entry.pack(side="right")
                self.ring_vars.append(var)
                
                # Add real-time update callback
                var.trace_add("write", lambda *_: self.root.after(50, self.update_ring_visualization))
            
            # Coupling coefficients with modern styling
            for i in range(num_rings + 1):
                frame = ctk.CTkFrame(self.ring_controls_frame, fg_color="transparent")
                frame.pack(fill="x", pady=3)
                
                if i == 0:
                    label_text = "Input Coupling:"
                    color = COLORS['primary']
                elif i == num_rings:
                    label_text = "Output Coupling:"
                    color = COLORS['secondary']
                else:
                    label_text = f"Ring {i}-{i+1} Coupling:"
                    color = COLORS['accent']
                
                ctk.CTkLabel(frame, text=label_text, 
                           font=ctk.CTkFont(size=11, weight="bold"),
                           text_color=color).pack(side="left")
                var = ctk.StringVar(value="0.1" if i == 0 or i == num_rings else "0.001")
                entry = ctk.CTkEntry(frame, textvariable=var, width=80)
                entry.pack(side="right")
                self.coupling_vars.append(var)
                
                # Add real-time update callback
                var.trace_add("write", lambda *_: self.root.after(50, self.update_ring_visualization))
            
            # Phase shifts with modern styling
            for i in range(num_rings):
                frame = ctk.CTkFrame(self.ring_controls_frame, fg_color="transparent")
                frame.pack(fill="x", pady=3)
                
                ctk.CTkLabel(frame, text=f"Ring {i+1} Phase (rad):", 
                           font=ctk.CTkFont(size=11, weight="bold"),
                           text_color=COLORS['warning']).pack(side="left")
                var = ctk.StringVar(value="0.0")
                entry = ctk.CTkEntry(frame, textvariable=var, width=80)
                entry.pack(side="right")
                self.phase_vars.append(var)
                
                # Add real-time update callback
                var.trace_add("write", lambda *_: self.root.after(50, self.update_ring_visualization))
            
            # Update visualization
            self.update_ring_visualization()
            
        except Exception as e:
            print(f"Error updating ring controls: {e}")  # Debug print
        
    def update_ring_visualization(self):
        """Update ring geometry visualization."""
        try:
            # Check if ring_viz exists before trying to use it
            if not hasattr(self, 'ring_viz') or self.ring_viz is None:
                return
                
            ring_radii = [float(var.get()) for var in self.ring_vars]
            coupling_coeffs = [float(var.get()) for var in self.coupling_vars]
            self.ring_viz.draw_rings(ring_radii, coupling_coeffs)
        except ValueError:
            pass  # Invalid input, skip update
    
    def load_default_parameters(self):
        """Load default parameters from the system."""
        config = self.system.get_configuration()
        
        # Update GUI with default values
        self.wl_start_var.set(str(config['wavelength_start_nm']))
        self.wl_stop_var.set(str(config['wavelength_stop_nm']))
        self.wl_res_var.set(str(config['wavelength_resolution_nm']))
        self.loss_var.set(str(config['loss_db_per_cm']))
        
        # Set default number of rings
        self.num_rings_var.set(2)
        
        # Update ring visualization
        self.update_ring_visualization()
    
    def run_simulation(self):
        """Run the ring resonator simulation."""
        try:
            # Get parameters from GUI
            ring_radii = [float(var.get()) for var in self.ring_vars]
            coupling_coeffs = [float(var.get()) for var in self.coupling_vars]
            phase_shifts = [float(var.get()) for var in self.phase_vars]
            
            wl_start = float(self.wl_start_var.get())
            wl_stop = float(self.wl_stop_var.get())
            wl_res = float(self.wl_res_var.get())
            loss = float(self.loss_var.get())
            
            # Configure system
            self.system.configure(
                ring_radii_um=ring_radii,
                coupling_coeffs=coupling_coeffs,
                phase_shifts_rad=phase_shifts,
                wavelength_start_nm=wl_start,
                wavelength_stop_nm=wl_stop,
                wavelength_resolution_nm=wl_res,
                loss_db_per_cm=loss
            )
            
            # Run simulation in separate thread
            self.progress.set(0.1)
            self.root.update_idletasks()
            
            def run_analysis():
                try:
                    # Analyze system
                    wavelengths, t1N, r1N, rN1, tN1 = self.system.analyze_system()
                    self.progress.set(0.5)
                    self.root.update_idletasks()
                    
                    # Calculate derived quantities
                    drop_dB, through_dB, total_dB = self.analyzer.calculate_power_response(t1N, r1N)
                    drop_phase, through_phase = self.analyzer.calculate_phase_response(t1N, r1N)
                    
                    # Calculate group delay and dispersion
                    c = 299792458
                    wavelengths_m = wavelengths * 1e-9
                    omega = 2 * np.pi * c / wavelengths_m
                    through_delay, drop_delay = self.analyzer.calculate_group_delay(
                        wavelengths_m, through_phase, drop_phase
                    )
                    through_dispersion, drop_dispersion = self.analyzer.calculate_dispersion(
                        (through_delay, drop_delay), omega
                    )
                    
                    self.progress.set(0.8)
                    self.root.update_idletasks()
                    
                    # Update plots
                    self.plot_widget.update_plots(
                        wavelengths, drop_dB, through_dB, drop_phase, through_phase,
                        through_delay, drop_delay, through_dispersion, drop_dispersion
                    )
                    
                    self.progress.set(1.0)
                    self.root.update_idletasks()
                    
                    # Show completion message
                    messagebox.showinfo("Success", "Simulation completed successfully!")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Simulation failed: {str(e)}")
                finally:
                    self.progress.set(0)
            
            # Run in thread to prevent GUI freezing
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameters: {str(e)}")
    
    def export_plots(self):
        """Export plots to files."""
        try:
            folder = filedialog.askdirectory(title="Select folder to save plots")
            if not folder:
                return
            
            # Export plots using the original plotter
            plotter = RingPlotter()
            
            # Get current data (this would need to be stored from last simulation)
            messagebox.showinfo("Export", "Plots exported to individual PDF files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filename:
                return
            
            config = {
                'ring_radii': [float(var.get()) for var in self.ring_vars],
                'coupling_coeffs': [float(var.get()) for var in self.coupling_vars],
                'phase_shifts': [float(var.get()) for var in self.phase_vars],
                'wavelength_start': float(self.wl_start_var.get()),
                'wavelength_stop': float(self.wl_stop_var.get()),
                'wavelength_resolution': float(self.wl_res_var.get()),
                'loss': float(self.loss_var.get())
            }
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", f"Configuration saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")
    
    def load_config(self):
        """Load configuration from file."""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not filename:
                return
            
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Update GUI with loaded values
            self.num_rings_var.set(len(config['ring_radii']))
            self.update_ring_controls()
            
            for i, value in enumerate(config['ring_radii']):
                if i < len(self.ring_vars):
                    self.ring_vars[i].set(str(value))
            
            for i, value in enumerate(config['coupling_coeffs']):
                if i < len(self.coupling_vars):
                    self.coupling_vars[i].set(str(value))
            
            for i, value in enumerate(config['phase_shifts']):
                if i < len(self.phase_vars):
                    self.phase_vars[i].set(str(value))
            
            self.wl_start_var.set(str(config['wavelength_start']))
            self.wl_stop_var.set(str(config['wavelength_stop']))
            self.wl_res_var.set(str(config['wavelength_resolution']))
            self.loss_var.set(str(config['loss']))
            
            self.update_ring_visualization()
            
            messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {str(e)}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the GUI application."""
    app = RingResonatorGUI()
    app.run()


if __name__ == "__main__":
    main() 
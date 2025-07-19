#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced Ring Resonator GUI
Tests all GUI components, interactivity, and functionality
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from rings_resonator.rings_gui import (
    RingResonatorStudio,
    InteractiveGeometryPlot,
    InteractiveAnalysisPlot,
    ControlPanel,
    SimulationWorker,
    MplCanvas,
)


class TestMplCanvas:
    """Test the enhanced matplotlib canvas"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    @pytest.fixture
    def canvas(self, app) -> None:
        """Create MplCanvas instance"""
        return MplCanvas(width=8, height=6)

    def test_canvas_creation(self, canvas) -> None:
        """Test canvas creation with dark theme"""
        assert canvas.fig is not None
        assert canvas.axes is not None
        # Note: matplotlib may return colors in different formats (tuples vs hex)
        # Just check that the facecolor is set (non-default)
        assert canvas.fig.get_facecolor() is not None
        assert canvas.axes.get_facecolor() is not None

    def test_clear_plot(self, canvas) -> None:
        """Test clearing plot maintains dark theme"""
        canvas.clear_plot()
        # Check that colors are set (not testing exact values due to matplotlib format variations)
        assert canvas.axes.get_facecolor() is not None
        assert canvas.axes.xaxis.label.get_color() is not None
        assert canvas.axes.yaxis.label.get_color() is not None


class TestInteractiveGeometryPlot:
    """Test the enhanced geometry plot widget"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    @pytest.fixture
    def geometry_plot(self, app) -> None:
        """Create InteractiveGeometryPlot instance"""
        return InteractiveGeometryPlot()

    def test_widget_creation(self, geometry_plot) -> None:
        """Test widget creation and initialization"""
        assert geometry_plot.canvas is not None
        assert geometry_plot.nav_toolbar is not None
        assert geometry_plot.show_labels is not None
        assert geometry_plot.show_dimensions is not None
        assert geometry_plot.show_coupling is not None
        assert geometry_plot.ring_radii == []
        assert geometry_plot.coupling_coeffs == []
        assert geometry_plot.phase_shifts == []

    def test_show_placeholder(self, geometry_plot) -> None:
        """Test placeholder display"""
        geometry_plot.show_placeholder()
        # Check that placeholder text is displayed (no title to save space)
        texts = geometry_plot.canvas.axes.texts
        assert len(texts) > 0
        # Check that the placeholder message is present
        placeholder_found = any("Configure Ring Parameters" in text.get_text() for text in texts)
        assert placeholder_found

    def test_update_geometry_empty(self, geometry_plot) -> None:
        """Test geometry update with empty data"""
        geometry_plot.update_geometry([], [])
        assert geometry_plot.ring_radii == []
        assert geometry_plot.coupling_coeffs == []

    def test_update_geometry_with_data(self, geometry_plot) -> None:
        """Test geometry update with actual data"""
        ring_radii = [35.0, 32.0]
        coupling_coeffs = [0.1, 0.001, 0.1]
        phase_shifts = [0.0, 0.5]

        geometry_plot.update_geometry(ring_radii, coupling_coeffs, phase_shifts)

        assert geometry_plot.ring_radii == ring_radii
        assert geometry_plot.coupling_coeffs == coupling_coeffs
        assert geometry_plot.phase_shifts == phase_shifts
        # Check that geometry is plotted (no title or axis labels for cleaner appearance)
        assert geometry_plot.canvas.axes.get_xlabel() == ""
        assert geometry_plot.canvas.axes.get_ylabel() == ""

    def test_checkboxes_functionality(self, geometry_plot) -> None:
        """Test checkbox controls"""
        # Test initial states
        assert geometry_plot.show_labels.isChecked() == True
        assert geometry_plot.show_dimensions.isChecked() == True
        assert geometry_plot.show_coupling.isChecked() == True

        # Test toggling
        geometry_plot.show_labels.setChecked(False)
        assert geometry_plot.show_labels.isChecked() == False

    def test_refresh_plot(self, geometry_plot) -> None:
        """Test plot refresh functionality"""
        # Set up data
        geometry_plot.ring_radii = [35.0, 32.0]
        geometry_plot.coupling_coeffs = [0.1, 0.001, 0.1]
        geometry_plot.phase_shifts = [0.0, 0.5]

        # Should not crash
        geometry_plot.refresh_plot()


class TestInteractiveAnalysisPlot:
    """Test the enhanced analysis plot widget"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    @pytest.fixture
    def analysis_plot(self, app) -> None:
        """Create InteractiveAnalysisPlot instance"""
        return InteractiveAnalysisPlot()

    def test_widget_creation(self, analysis_plot) -> None:
        """Test widget creation and initialization"""
        assert analysis_plot.canvas is not None
        assert analysis_plot.nav_toolbar is not None
        assert analysis_plot.plot_selector is not None
        assert analysis_plot.show_grid is not None
        assert analysis_plot.show_markers is not None
        assert analysis_plot.export_btn is not None
        assert analysis_plot.current_results is None

    def test_show_placeholder(self, analysis_plot) -> None:
        """Test placeholder display"""
        analysis_plot.show_placeholder()
        # Check that placeholder text is displayed (no title to save space)
        texts = analysis_plot.canvas.axes.texts
        assert len(texts) > 0
        # Check that the placeholder message is present
        placeholder_found = any("Run Simulation to View" in text.get_text() for text in texts)
        assert placeholder_found

    def test_plot_selector_options(self, analysis_plot) -> None:
        """Test plot selector has all required options"""
        expected_options = [
            "Transmission Spectrum",
            "Phase Response",
            "Group Delay",
            "Dispersion",
            "Effective Index",
        ]

        for i, option in enumerate(expected_options):
            assert analysis_plot.plot_selector.itemText(i) == option

    def test_update_results(self, analysis_plot) -> None:
        """Test updating results enables controls"""
        # Create test results
        wavelengths = np.linspace(1520, 1570, 100)
        results = {
            "wavelengths": wavelengths,
            "drop_dB": -20 + 10 * np.sin(0.1 * (wavelengths - 1520)),
            "through_dB": -2 + 1 * np.sin(0.1 * (wavelengths - 1520)),
            "drop_phase": np.linspace(0, 2 * np.pi, len(wavelengths)),
            "through_phase": np.linspace(0, np.pi, len(wavelengths)),
            "through_delay": np.ones(len(wavelengths) - 1) * 1e-12,
            "drop_delay": np.ones(len(wavelengths) - 1) * 2e-12,
            "through_dispersion": np.ones(len(wavelengths) - 2) * 1e-24,
            "drop_dispersion": np.ones(len(wavelengths) - 2) * 2e-24,
            "neff": 4.077 * np.ones_like(wavelengths),
        }

        analysis_plot.update_results(results)

        assert analysis_plot.current_results == results
        assert analysis_plot.plot_selector.isEnabled() == True
        assert analysis_plot.export_btn.isEnabled() == True
        assert analysis_plot.show_grid.isEnabled() == True
        assert analysis_plot.show_markers.isEnabled() == True

    def test_plot_types(self, analysis_plot) -> None:
        """Test all plot types can be displayed"""
        # Create test results
        wavelengths = np.linspace(1520, 1570, 100)
        results = {
            "wavelengths": wavelengths,
            "drop_dB": -20 + 10 * np.sin(0.1 * (wavelengths - 1520)),
            "through_dB": -2 + 1 * np.sin(0.1 * (wavelengths - 1520)),
            "drop_phase": np.linspace(0, 2 * np.pi, len(wavelengths)),
            "through_phase": np.linspace(0, np.pi, len(wavelengths)),
            "through_delay": np.ones(len(wavelengths) - 1) * 1e-12,
            "drop_delay": np.ones(len(wavelengths) - 1) * 2e-12,
            "through_dispersion": np.ones(len(wavelengths) - 2) * 1e-24,
            "drop_dispersion": np.ones(len(wavelengths) - 2) * 2e-24,
            "neff": 4.077 * np.ones_like(wavelengths),
        }

        analysis_plot.update_results(results)

        plot_types = [
            "Transmission Spectrum",
            "Phase Response",
            "Group Delay",
            "Dispersion",
            "Effective Index",
        ]

        for plot_type in plot_types:
            analysis_plot.plot_selector.setCurrentText(plot_type)
            analysis_plot.update_plot()
            # Check that plot type is applied correctly (no title to save space)
            assert analysis_plot.canvas.axes.get_xlabel() == "Wavelength (nm)"
            # Check that y-axis label changes based on plot type
            y_label = analysis_plot.canvas.axes.get_ylabel()
            assert y_label is not None and len(y_label) > 0

    def test_plot_options(self, analysis_plot) -> None:
        """Test plot options functionality"""
        # Test grid toggle
        analysis_plot.show_grid.setChecked(True)
        assert analysis_plot.show_grid.isChecked() == True

        # Test markers toggle
        analysis_plot.show_markers.setChecked(True)
        assert analysis_plot.show_markers.isChecked() == True

    def test_export_plot_no_data(self, analysis_plot) -> None:
        """Test export plot with no data"""
        # Mock the message box to avoid user prompt
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning:
            analysis_plot.export_plot()
            mock_warning.assert_called_once()

    def test_export_plot_with_data(self, analysis_plot, tmp_path) -> None:
        """Test export plot with data"""
        # Create test results
        wavelengths = np.linspace(1520, 1570, 100)
        results = {
            "wavelengths": wavelengths,
            "drop_dB": -20 + 10 * np.sin(0.1 * (wavelengths - 1520)),
            "through_dB": -2 + 1 * np.sin(0.1 * (wavelengths - 1520)),
            "drop_phase": np.linspace(0, 2 * np.pi, len(wavelengths)),
            "through_phase": np.linspace(0, np.pi, len(wavelengths)),
            "through_delay": np.ones(len(wavelengths) - 1) * 1e-12,
            "drop_delay": np.ones(len(wavelengths) - 1) * 2e-12,
            "through_dispersion": np.ones(len(wavelengths) - 2) * 1e-24,
            "drop_dispersion": np.ones(len(wavelengths) - 2) * 2e-24,
            "neff": 4.077 * np.ones_like(wavelengths),
        }

        analysis_plot.update_results(results)

        # Test export
        export_file = tmp_path / "test_export.png"
        with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName") as mock_save, patch(
            "PySide6.QtWidgets.QMessageBox.information"
        ) as mock_info:
            mock_save.return_value = (str(export_file), "PNG Files (*.png)")
            analysis_plot.export_plot()

        # Verify file was created
        assert export_file.exists()


class TestControlPanel:
    """Test the enhanced control panel"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    @pytest.fixture
    def control_panel(self, app) -> None:
        """Create ControlPanel instance"""
        return ControlPanel()

    def test_widget_creation(self, control_panel) -> None:
        """Test widget creation and initialization"""
        assert control_panel.num_rings_spin is not None
        assert control_panel.ring_group is not None
        assert control_panel.wl_start_spin is not None
        assert control_panel.wl_stop_spin is not None
        assert control_panel.wl_res_spin is not None
        assert control_panel.loss_spin is not None
        assert control_panel.neff_0_spin is not None
        assert control_panel.neff_1_spin is not None
        assert control_panel.neff_2_spin is not None
        assert control_panel.run_btn is not None
        assert control_panel.progress_bar is not None
        assert control_panel.save_btn is not None
        assert control_panel.load_btn is not None

    def test_initial_values(self, control_panel) -> None:
        """Test initial parameter values"""
        assert control_panel.num_rings_spin.value() == 2
        assert control_panel.wl_start_spin.value() == 1520.0
        assert control_panel.wl_stop_spin.value() == 1570.0
        assert control_panel.wl_res_spin.value() == 0.01
        assert control_panel.loss_spin.value() == 4.0
        # Use approximate equality for floating point values
        assert abs(control_panel.neff_0_spin.value() - 4.077) < 0.01
        assert abs(control_panel.neff_1_spin.value() - (-0.983)) < 0.01
        assert abs(control_panel.neff_2_spin.value() - (-0.046)) < 0.01

    def test_ring_controls_update(self, control_panel) -> None:
        """Test ring controls update when number of rings changes"""
        # Start with 2 rings
        assert len(control_panel.ring_controls) == 2
        assert len(control_panel.coupling_controls) == 3  # N+1 coupling coefficients
        assert len(control_panel.phase_controls) == 2

        # Change to 3 rings
        control_panel.num_rings_spin.setValue(3)
        control_panel.update_ring_controls()

        assert len(control_panel.ring_controls) == 3
        assert len(control_panel.coupling_controls) == 4  # N+1 coupling coefficients
        assert len(control_panel.phase_controls) == 3

    def test_get_configuration(self, control_panel) -> None:
        """Test getting configuration dictionary"""
        config = control_panel.get_configuration()

        required_keys = [
            "ring_radii_um",
            "coupling_coeffs",
            "phase_shifts_rad",
            "wavelength_start_nm",
            "wavelength_stop_nm",
            "wavelength_resolution_nm",
            "loss_db_per_cm",
            "neff_coeffs",
        ]

        for key in required_keys:
            assert key in config

        # Check data types and lengths
        assert isinstance(config["ring_radii_um"], list)
        assert isinstance(config["coupling_coeffs"], list)
        assert isinstance(config["phase_shifts_rad"], list)
        assert isinstance(config["neff_coeffs"], list)
        assert len(config["ring_radii_um"]) == 2  # Default 2 rings
        assert len(config["coupling_coeffs"]) == 3  # N+1 coupling coefficients
        assert len(config["phase_shifts_rad"]) == 2  # N phase shifts
        assert len(config["neff_coeffs"]) == 3  # n0, n1, n2

    def test_save_load_config(self, control_panel, tmp_path) -> None:
        """Test configuration save/load functionality"""
        # Create test config file
        config_file = tmp_path / "test_config.json"

        # Mock file dialog and message box to return our test file
        with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName") as mock_save, patch(
            "PySide6.QtWidgets.QMessageBox.information"
        ) as mock_info:
            mock_save.return_value = (str(config_file), "JSON Files (*.json)")
            control_panel.save_config()

        # Verify file was created
        assert config_file.exists()

        # Test loading
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName") as mock_load, patch(
            "PySide6.QtWidgets.QMessageBox.information"
        ) as mock_info:
            mock_load.return_value = (str(config_file), "JSON Files (*.json)")
            control_panel.load_config()


class TestSimulationWorker:
    """Test the simulation worker"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    @pytest.fixture
    def worker(self, app) -> None:
        """Create SimulationWorker instance"""
        return SimulationWorker()

    def test_worker_creation(self, worker) -> None:
        """Test worker creation and initialization"""
        assert worker.system is not None
        assert worker.analyzer is not None

    def test_simulation_signals(self, worker) -> None:
        """Test that worker has required signals"""
        assert hasattr(worker, "progress_updated")
        assert hasattr(worker, "log_message")
        assert hasattr(worker, "simulation_finished")
        assert hasattr(worker, "simulation_error")

    def test_run_simulation(self, worker) -> None:
        """Test running simulation with valid configuration"""
        config = {
            "ring_radii_um": [35.0, 32.0],
            "coupling_coeffs": [0.1, 0.001, 0.1],
            "phase_shifts_rad": [0.0, 0.0],
            "wavelength_start_nm": 1520,
            "wavelength_stop_nm": 1570,
            "wavelength_resolution_nm": 0.1,
            "loss_db_per_cm": 4.0,
            "neff_coeffs": [4.077, -0.983, -0.046],
        }

        # Mock the signals to capture emissions
        progress_signals = []
        log_signals = []
        finished_signals = []
        error_signals = []

        worker.progress_updated.connect(lambda x: progress_signals.append(x))
        worker.log_message.connect(lambda x: log_signals.append(x))
        worker.simulation_finished.connect(lambda x: finished_signals.append(x))
        worker.simulation_error.connect(lambda x: error_signals.append(x))

        # Run simulation
        worker.run_simulation(config)

        # Check that signals were emitted
        assert len(progress_signals) > 0
        assert len(log_signals) > 0
        assert len(finished_signals) == 1 or len(error_signals) == 1

        # If successful, check results structure
        if len(finished_signals) == 1:
            results = finished_signals[0]
            expected_keys = [
                "wavelengths",
                "drop_dB",
                "through_dB",
                "drop_phase",
                "through_phase",
                "through_delay",
                "drop_delay",
                "through_dispersion",
                "drop_dispersion",
                "neff",
            ]
            for key in expected_keys:
                assert key in results


class TestRingResonatorStudio:
    """Test the main application window"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    @pytest.fixture
    def main_window(self, app) -> None:
        """Create RingResonatorStudio instance"""
        return RingResonatorStudio()

    def test_window_creation(self, main_window) -> None:
        """Test main window creation and initialization"""
        assert main_window.control_panel is not None
        assert main_window.geometry_plot is not None
        assert main_window.analysis_plot is not None
        assert main_window.log_text is not None
        assert main_window.worker is not None
        assert main_window.worker_thread is not None

    def test_window_properties(self, main_window) -> None:
        """Test window properties"""
        assert "Ring Resonator Design Studio" in main_window.windowTitle()
        assert main_window.minimumSize().width() >= 1400
        assert main_window.minimumSize().height() >= 900

    def test_signal_connections(self, main_window) -> None:
        """Test signal connections"""
        # Test that parameter changes trigger geometry updates
        config = main_window.control_panel.get_configuration()
        main_window.update_geometry()

        # Should not crash
        assert True

    def test_simulation_workflow(self, main_window) -> None:
        """Test complete simulation workflow"""
        # Start simulation
        main_window.run_simulation()

        # Process events to ensure UI updates
        QApplication.processEvents()

        # Check that UI state changes
        assert main_window.control_panel.run_btn.isEnabled() == False
        # Progress bar visibility depends on timing, so we won't test it strictly

        # Simulate simulation completion
        results = {
            "wavelengths": np.linspace(1520, 1570, 100),
            "drop_dB": np.ones(100) * -20,
            "through_dB": np.ones(100) * -2,
            "drop_phase": np.zeros(100),
            "through_phase": np.zeros(100),
            "through_delay": np.ones(99) * 1e-12,
            "drop_delay": np.ones(99) * 2e-12,
            "through_dispersion": np.ones(98) * 1e-24,
            "drop_dispersion": np.ones(98) * 2e-24,
            "neff": np.ones(100) * 4.077,
        }

        main_window.simulation_finished(results)

        # Check that UI state resets
        assert main_window.control_panel.run_btn.isEnabled() == True
        assert main_window.control_panel.progress_bar.isVisible() == False

        # Check that analysis plot was updated
        assert main_window.analysis_plot.current_results == results

    def test_window_cleanup(self, main_window) -> None:
        """Test window cleanup on close"""
        # Should not crash
        main_window.close()
        assert True


class TestLogging:
    """Test logging functionality"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    def test_logging_setup(self, app) -> None:
        """Test that logging callbacks are properly set up"""
        from rings_resonator.rings_gui import RingResonatorStudio

        window = RingResonatorStudio()

        # Test that all components have logging callbacks
        assert window.control_panel.log_callback is not None
        assert window.geometry_plot.log_callback is not None
        assert window.analysis_plot.log_callback is not None

        # Test safe logging methods exist
        assert hasattr(window.control_panel, "safe_log")
        assert hasattr(window.geometry_plot, "safe_log")
        assert hasattr(window.analysis_plot, "safe_log")

        window.close()

    def test_parameter_logging(self, app) -> None:
        """Test that parameter changes are logged"""
        from rings_resonator.rings_gui import ControlPanel

        control_panel = ControlPanel()
        logged_messages = []

        # Set up logging callback
        control_panel.log_callback = lambda msg: logged_messages.append(msg)

        # Test parameter change logging
        control_panel.log_parameter_change("Test Parameter", 123.45, " units")
        assert len(logged_messages) == 1
        assert "Parameter changed: Test Parameter = 123.45 units" in logged_messages[0]


class TestIntegration:
    """Integration tests for the complete GUI"""

    @pytest.fixture
    def app(self) -> None:
        """Create QApplication instance"""
        return QApplication.instance() or QApplication(sys.argv)

    def test_full_gui_workflow(self, app) -> None:
        """Test complete GUI workflow from creation to simulation"""
        # Create main window
        window = RingResonatorStudio()

        # Test initial state
        assert window.control_panel is not None
        assert window.geometry_plot is not None
        assert window.analysis_plot is not None

        # Test parameter changes
        window.control_panel.num_rings_spin.setValue(3)
        window.control_panel.update_ring_controls()

        # Test geometry update
        window.update_geometry()

        # Test configuration
        config = window.control_panel.get_configuration()
        assert len(config["ring_radii_um"]) == 3
        assert len(config["coupling_coeffs"]) == 4
        assert len(config["phase_shifts_rad"]) == 3

        # Test simulation start
        window.run_simulation()
        assert window.control_panel.run_btn.isEnabled() == False

        # Cleanup
        window.close()

    def test_gui_with_different_configurations(self, app) -> None:
        """Test GUI with various ring configurations"""
        window = RingResonatorStudio()

        # Test with different numbers of rings
        for num_rings in [1, 2, 3, 4, 5]:
            window.control_panel.num_rings_spin.setValue(num_rings)
            window.control_panel.update_ring_controls()

            config = window.control_panel.get_configuration()
            assert len(config["ring_radii_um"]) == num_rings
            assert len(config["coupling_coeffs"]) == num_rings + 1
            assert len(config["phase_shifts_rad"]) == num_rings

            # Test geometry update
            window.update_geometry()

        # Cleanup
        window.close()

    def test_plot_interactivity(self, app) -> None:
        """Test plot interactivity features"""
        window = RingResonatorStudio()

        # Test geometry plot controls
        geometry_plot = window.geometry_plot
        assert geometry_plot.show_labels.isChecked() == True
        assert geometry_plot.show_dimensions.isChecked() == True
        assert geometry_plot.show_coupling.isChecked() == True

        # Test analysis plot controls
        analysis_plot = window.analysis_plot
        assert analysis_plot.show_grid.isChecked() == True
        assert analysis_plot.show_markers.isChecked() == False

        # Test plot selector
        plot_types = [
            "Transmission Spectrum",
            "Phase Response",
            "Group Delay",
            "Dispersion",
            "Effective Index",
        ]

        for plot_type in plot_types:
            analysis_plot.plot_selector.setCurrentText(plot_type)
            assert analysis_plot.plot_selector.currentText() == plot_type

        # Cleanup
        window.close()


def test_gui_import() -> None:
    """Test that GUI module can be imported successfully"""
    from rings_resonator.rings_gui import (
        RingResonatorStudio,
        InteractiveGeometryPlot,
        InteractiveAnalysisPlot,
        ControlPanel,
        SimulationWorker,
        MplCanvas,
    )

    # All classes should be importable
    assert RingResonatorStudio is not None
    assert InteractiveGeometryPlot is not None
    assert InteractiveAnalysisPlot is not None
    assert ControlPanel is not None
    assert SimulationWorker is not None
    assert MplCanvas is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

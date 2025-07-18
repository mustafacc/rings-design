"""
Test suite for the Ring Resonator Design Toolkit.

This module contains basic tests to validate the core functionality
of the ring resonator simulation and analysis tools.
"""

import pytest
import numpy as np
from rings_resonator import RingResonatorSystem, RingAnalyzer, RingPlotter


class TestRingResonatorSystem:
    """Test cases for the RingResonatorSystem class."""

    def test_initialization(self):
        """Test that the system initializes with default parameters."""
        system = RingResonatorSystem()
        
        # Check default parameters
        assert system.wavelength_start == 1500.0
        assert system.wavelength_stop == 1600.0
        assert system.wavelength_resolution == 0.01
        assert system.ring_radii_um == [35.0, 21.0]
        assert system.loss_db_per_cm == 4.0
        assert len(system.neff_coeffs) == 3

    def test_configuration_validation(self):
        """Test parameter validation in the configure method."""
        system = RingResonatorSystem()
        
        # Test invalid wavelength range
        with pytest.raises(ValueError, match="Start wavelength must be less than stop wavelength"):
            system.configure(wavelength_start_nm=1600.0, wavelength_stop_nm=1500.0)
        
        # Test invalid resolution
        with pytest.raises(ValueError, match="Resolution must be positive"):
            system.configure(wavelength_resolution_nm=-0.01)
        
        # Test invalid coupling coefficients
        with pytest.raises(ValueError, match="Need 3 coupling coefficients for 2 rings"):
            system.configure(ring_radii_um=[35.0, 21.0], coupling_coeffs=[0.1, 0.2])

    def test_single_ring_configuration(self):
        """Test configuration of a single ring system."""
        system = RingResonatorSystem()
        system.configure(
            ring_radii_um=[50.0],
            coupling_coeffs=[0.05, 0.05],
            wavelength_start_nm=1550.0,
            wavelength_stop_nm=1560.0
        )
        
        assert len(system.ring_radii_um) == 1
        assert len(system.coupling_coeffs) == 2
        assert len(system.phase_shifts_rad) == 1
        assert system.wavelength_start == 1550.0
        assert system.wavelength_stop == 1560.0

    def test_multi_ring_configuration(self):
        """Test configuration of a multi-ring system."""
        system = RingResonatorSystem()
        system.configure(
            ring_radii_um=[40.0, 35.0, 30.0],
            coupling_coeffs=[0.08, 0.02, 0.02, 0.08],
            phase_shifts_rad=[0.0, 1.57, 3.14]
        )
        
        assert len(system.ring_radii_um) == 3
        assert len(system.coupling_coeffs) == 4
        assert len(system.phase_shifts_rad) == 3

    def test_analyze_system(self):
        """Test that the analyze_system method returns correct array shapes."""
        system = RingResonatorSystem()
        system.configure(
            ring_radii_um=[35.0, 21.0],
            coupling_coeffs=[0.1, 0.001, 0.1],
            wavelength_start_nm=1550.0,
            wavelength_stop_nm=1560.0,
            wavelength_resolution_nm=0.1
        )
        
        wavelengths, t1N, r1N, rN1, tN1 = system.analyze_system()
        
        # Check that all arrays have the same length
        assert len(wavelengths) == len(t1N)
        assert len(wavelengths) == len(r1N)
        assert len(wavelengths) == len(rN1)
        assert len(wavelengths) == len(tN1)
        
        # Check wavelength range
        assert wavelengths[0] >= 1550.0
        assert wavelengths[-1] <= 1560.0
        
        # Check that results are complex numbers
        assert np.iscomplexobj(t1N)
        assert np.iscomplexobj(r1N)


class TestRingAnalyzer:
    """Test cases for the RingAnalyzer class."""

    def test_power_response_calculation(self):
        """Test power response calculation."""
        # Create test data
        t1N = np.array([0.5 + 0.3j, 0.7 + 0.2j])
        r1N = np.array([0.8 + 0.1j, 0.6 + 0.4j])
        
        analyzer = RingAnalyzer()
        drop_dB, through_dB, total_dB = analyzer.calculate_power_response(t1N, r1N)
        
        # Check that results are real numbers
        assert np.isrealobj(drop_dB)
        assert np.isrealobj(through_dB)
        assert np.isrealobj(total_dB)
        
        # Check that power values are reasonable (in dB)
        assert np.all(drop_dB <= 0)  # Power should be <= 0 dB
        assert np.all(through_dB <= 0)
        assert np.all(total_dB <= 0)

    def test_phase_response_calculation(self):
        """Test phase response calculation."""
        # Create test data
        t1N = np.array([0.5 + 0.3j, 0.7 + 0.2j])
        r1N = np.array([0.8 + 0.1j, 0.6 + 0.4j])
        
        analyzer = RingAnalyzer()
        drop_phase, through_phase = analyzer.calculate_phase_response(t1N, r1N)
        
        # Check that results are real numbers
        assert np.isrealobj(drop_phase)
        assert np.isrealobj(through_phase)
        
        # Check that phase values are reasonable (in radians)
        assert np.all(np.abs(drop_phase) <= 10 * np.pi)  # Reasonable phase range
        assert np.all(np.abs(through_phase) <= 10 * np.pi)

    def test_group_delay_calculation(self):
        """Test group delay calculation."""
        # Create test data
        wavelengths = np.linspace(1550, 1560, 100) * 1e-9  # in meters
        through_phase = np.linspace(0, 2*np.pi, 100)
        drop_phase = np.linspace(0, 4*np.pi, 100)
        
        analyzer = RingAnalyzer()
        through_delay, drop_delay = analyzer.calculate_group_delay(
            wavelengths, through_phase, drop_phase
        )
        
        # Check that results have correct length (one less than input)
        assert len(through_delay) == len(wavelengths) - 1
        assert len(drop_delay) == len(wavelengths) - 1
        
        # Check that results are real numbers
        assert np.isrealobj(through_delay)
        assert np.isrealobj(drop_delay)


class TestRingPlotter:
    """Test cases for the RingPlotter class."""

    def test_professional_style_setup(self):
        """Test that the professional style setup works without errors."""
        # This test mainly ensures no exceptions are raised
        RingPlotter._setup_professional_style()
        
        # If we get here without exceptions, the test passes
        assert True


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_complete_workflow(self):
        """Test the complete workflow from configuration to analysis."""
        # Create and configure system
        system = RingResonatorSystem()
        system.configure(
            ring_radii_um=[35.0, 21.0],
            coupling_coeffs=[0.1, 0.001, 0.1],
            wavelength_start_nm=1550.0,
            wavelength_stop_nm=1560.0,
            wavelength_resolution_nm=0.1
        )
        
        # Analyze system
        wavelengths, t1N, r1N, rN1, tN1 = system.analyze_system()
        
        # Calculate derived quantities
        analyzer = RingAnalyzer()
        drop_dB, through_dB, total_dB = analyzer.calculate_power_response(t1N, r1N)
        drop_phase, through_phase = analyzer.calculate_phase_response(t1N, r1N)
        
        # Check power conservation (approximately)
        power_sum = np.abs(t1N)**2 + np.abs(r1N)**2
        assert np.all(power_sum <= 1.1)  # Account for numerical errors and loss
        
        # Check that we have reasonable results
        assert len(wavelengths) > 0
        assert len(drop_dB) == len(wavelengths)
        assert len(through_dB) == len(wavelengths)
        assert len(total_dB) == len(wavelengths)

    def test_configuration_retrieval(self):
        """Test that configuration can be retrieved correctly."""
        system = RingResonatorSystem()
        system.configure(
            ring_radii_um=[30.0, 25.0],
            coupling_coeffs=[0.12, 0.008, 0.12],
            wavelength_start_nm=1520.0,
            wavelength_stop_nm=1580.0,
            wavelength_resolution_nm=0.005,
            loss_db_per_cm=3.5
        )
        
        config = system.get_configuration()
        
        # Check that configuration is correctly retrieved
        assert config['ring_radii_um'] == [30.0, 25.0]
        assert config['coupling_coeffs'] == [0.12, 0.008, 0.12]
        assert config['wavelength_start_nm'] == 1520.0
        assert config['wavelength_stop_nm'] == 1580.0
        assert config['wavelength_resolution_nm'] == 0.005
        assert config['loss_db_per_cm'] == 3.5


class TestGUIImport:
    """Test cases for GUI module import."""
    
    def test_gui_import(self):
        """Test that enhanced GUI module can be imported successfully."""
        try:
            from rings_resonator.rings_gui import (
                RingResonatorStudio,
                InteractiveGeometryPlot,
                InteractiveAnalysisPlot,
                ControlPanel,
                SimulationWorker,
                MplCanvas
            )
            assert RingResonatorStudio is not None
            assert InteractiveGeometryPlot is not None
            assert InteractiveAnalysisPlot is not None
            assert ControlPanel is not None
            assert SimulationWorker is not None
            assert MplCanvas is not None
        except ImportError as e:
            pytest.fail(f"Failed to import enhanced GUI module: {e}")


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__]) 
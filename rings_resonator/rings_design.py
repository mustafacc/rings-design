"""
Ring Resonator Designer and Simulator

This module provides tools for designing and simulating coupled ring resonator systems.
It calculates transmission spectra, phase response, group delay, and dispersion
characteristics for multi-ring configurations.

Author: Mustafa Hammood
Based on: Originally based on H. Shoman 2017 MATLAB implementation

The number of rings is controlled by the length of the R (radii) list:
- R = [r1, r2] creates a 2-ring system
- phi = [phi1, phi2] sets initial phases for each ring
- kappa = [k1, k2, k3] defines coupling coefficients:
  * k1: bus waveguide to first ring
  * k2: first ring to second ring  
  * k3: last ring to bus waveguide (for N rings, need N+1 couplers)
"""

import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
from typing import Tuple, List
import logging


class RingResonatorSystem:
    """
    A class to represent and analyze coupled ring resonator systems.
    
    All wavelengths are specified in nanometers for user convenience.
    All ring radii are specified in micrometers for typical silicon photonics scales.
    """
    
    def __init__(self):
        """Initialize the ring resonator system with default parameters."""
        self._set_default_parameters()
    
    def _set_default_parameters(self) -> None:
        """Set default system parameters."""
        # Wavelength range and resolution (nm)
        self.wavelength_start = 1500.0  # nm
        self.wavelength_stop = 1600.0   # nm  
        self.wavelength_resolution = 0.01  # nm
        
        # Waveguide effective index polynomial fit (for 500nm width waveguide)
        # neff = n1 + n2*λ(μm) + n3*λ(μm)²
        self.neff_coeffs = [4.077182700600432, -0.982556173493906, -0.046366956781710]
        
        # Waveguide loss in dB/cm
        self.loss_db_per_cm = 4.0
        
        # Ring configuration (micrometers)
        self.ring_radii_um = [35.0, 21.0]  # μm
        self.phase_shifts_rad = [0.0, 0.0]  # radians
        
        # Coupling coefficients (power coupling)
        self.coupling_coeffs = [0.2, 0.025, 0.2]  # [bus-ring1, ring1-ring2, ring2-bus]
    
    def configure(self, 
                  ring_radii_um: List[float] = None,
                  coupling_coeffs: List[float] = None,
                  phase_shifts_rad: List[float] = None,
                  wavelength_start_nm: float = None,
                  wavelength_stop_nm: float = None,
                  wavelength_resolution_nm: float = None,
                  loss_db_per_cm: float = None,
                  neff_coeffs: List[float] = None) -> None:
        """
        Configure all ring resonator system parameters.
        
        Args:
            ring_radii_um: List of ring radii in micrometers (e.g., [35.0, 21.0])
            coupling_coeffs: List of power coupling coefficients (length = num_rings + 1)
            phase_shifts_rad: List of phase shifts in radians (optional, defaults to zeros)
            wavelength_start_nm: Starting wavelength in nanometers (e.g., 1500.0)
            wavelength_stop_nm: Ending wavelength in nanometers (e.g., 1600.0)
            wavelength_resolution_nm: Wavelength resolution in nanometers (e.g., 0.01)
            loss_db_per_cm: Waveguide loss in dB/cm (e.g., 4.0)
            neff_coeffs: Effective index polynomial coefficients [n1, n2, n3]
        
        Raises:
            ValueError: If parameter dimensions are inconsistent
        """
        # Configure ring parameters
        if ring_radii_um is not None:
            self.ring_radii_um = ring_radii_um.copy()
            num_rings = len(self.ring_radii_um)
            
            # Validate coupling coefficients
            if coupling_coeffs is not None:
                if len(coupling_coeffs) != num_rings + 1:
                    raise ValueError(f"Need {num_rings + 1} coupling coefficients for {num_rings} rings")
                self.coupling_coeffs = coupling_coeffs.copy()
            
            # Configure phase shifts
            if phase_shifts_rad is not None:
                if len(phase_shifts_rad) != num_rings:
                    raise ValueError(f"Need {num_rings} phase shifts for {num_rings} rings")
                self.phase_shifts_rad = phase_shifts_rad.copy()
            else:
                self.phase_shifts_rad = [0.0] * num_rings
        
        elif coupling_coeffs is not None:
            self.coupling_coeffs = coupling_coeffs.copy()
        
        # Configure wavelength parameters
        if wavelength_start_nm is not None:
            self.wavelength_start = wavelength_start_nm
        if wavelength_stop_nm is not None:
            self.wavelength_stop = wavelength_stop_nm
        if wavelength_resolution_nm is not None:
            self.wavelength_resolution = wavelength_resolution_nm
            
        # Validate wavelength range
        if self.wavelength_start >= self.wavelength_stop:
            raise ValueError("Start wavelength must be less than stop wavelength")
        if self.wavelength_resolution <= 0:
            raise ValueError("Resolution must be positive")
        
        # Configure material parameters
        if loss_db_per_cm is not None:
            self.loss_db_per_cm = loss_db_per_cm
        if neff_coeffs is not None:
            if len(neff_coeffs) != 3:
                raise ValueError("Need exactly 3 effective index coefficients [n1, n2, n3]")
            self.neff_coeffs = neff_coeffs.copy()
        
        # Print configuration summary
        self._print_configuration()
    
    def _print_configuration(self) -> None:
        """Print current system configuration."""
        print("Ring Resonator System Configuration:")
        print(f"  Wavelength range: {self.wavelength_start:.1f} - {self.wavelength_stop:.1f} nm")
        print(f"  Resolution: {self.wavelength_resolution:.3f} nm")
        print(f"  Ring radii: {self.ring_radii_um} μm")
        print(f"  Coupling coefficients: {self.coupling_coeffs}")
        print(f"  Phase shifts: {self.phase_shifts_rad} rad")
        print(f"  Loss: {self.loss_db_per_cm:.1f} dB/cm")
        print(f"  Effective index coeffs: {self.neff_coeffs}")
    
    def get_configuration(self) -> dict:
        """
        Get current system configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'wavelength_start_nm': self.wavelength_start,
            'wavelength_stop_nm': self.wavelength_stop,
            'wavelength_resolution_nm': self.wavelength_resolution,
            'ring_radii_um': self.ring_radii_um.copy(),
            'coupling_coeffs': self.coupling_coeffs.copy(),
            'phase_shifts_rad': self.phase_shifts_rad.copy(),
            'loss_db_per_cm': self.loss_db_per_cm,
            'neff_coeffs': self.neff_coeffs.copy()
        }
    
    def _calculate_ring_transfer_matrix(self, kappa: float, phi: float, 
                                     circumference: float, beta: float, 
                                     alpha: float) -> List[List[complex]]:
        """
        Calculate the scattering matrix for a single ring resonator.
        
        Args:
            kappa: Power coupling coefficient
            phi: Phase shift in radians
            circumference: Ring circumference in meters
            beta: Propagation constant in rad/m
            alpha: Loss coefficient in dB/m
            
        Returns:
            2x2 scattering matrix as nested list
        """
        j = 1j
        
        # Loss factor (field amplitude)
        gamma = 10**(-alpha * circumference / 20)
        
        # Round-trip phase accumulation
        zeta = gamma * np.exp(-j * phi) * np.exp(-j * beta * circumference)
        sqrt_zeta = math.sqrt(gamma) * np.exp(-j * phi / 2) * np.exp(-j * beta * circumference / 2)
        
        # Coupling coefficients (field)
        s = math.sqrt(kappa)      # coupling coefficient
        c = math.sqrt(1 - kappa)  # transmission coefficient
        
        # Scattering matrix elements
        t1N = -j * s * sqrt_zeta  # through transmission
        rN1 = c * zeta           # drop reflection  
        r1N = c                  # through reflection
        tN1 = t1N               # drop transmission
        
        return [[t1N, rN1], [r1N, tN1]]
    
    def _scattering_to_transfer_matrix(self, S: np.ndarray) -> np.ndarray:
        """Convert scattering matrix to transfer matrix."""
        A = S[0,0] * S[1,1] - S[0,1] * S[1,0]
        B = S[0,1]
        C = -S[1,0] 
        D = 1.0
        
        M_temp = np.array([[A, B], [C, D]], dtype=complex)
        return (1 / S[1,1]) * M_temp
    
    def _transfer_to_scattering_matrix(self, M: np.ndarray) -> np.ndarray:
        """Convert transfer matrix to scattering matrix."""
        t1N = M[0,0] * M[1,1] - M[0,1] * M[1,0]
        rN1 = M[0,1]
        r1N = -M[1,0]
        tN1 = 1.0
        
        S_temp = np.array([[t1N, rN1], [r1N, tN1]], dtype=complex)
        return (1 / M[1,1]) * S_temp
    
    def analyze_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform theoretical analysis of the ring resonator system.
        Matches the exact algorithm from the original MATLAB-based script.
        
        Returns:
            Tuple containing:
            - wavelengths: Wavelength array in nanometers
            - t1N: Drop port field transmission coefficients
            - r1N: Through port field transmission coefficients  
            - rN1: Drop port field reflection coefficients
            - tN1: Through port field reflection coefficients
        """
        logger = logging.getLogger('RingResonatorGUI')
        
        # Generate wavelength array in nanometers - exact match to original
        # Original: lambda_0 = np.linspace(wavelength_start, wavelength_stop, round((wavelength_stop-wavelength_start)*1e9/resolution))
        # where wavelength_start/stop are in meters and resolution is in nm
        wavelength_start_m = self.wavelength_start * 1e-9
        wavelength_stop_m = self.wavelength_stop * 1e-9
        num_points = round((wavelength_stop_m - wavelength_start_m) * 1e9 / self.wavelength_resolution)
        wavelengths_nm = np.linspace(self.wavelength_start, self.wavelength_stop, num_points)
        
        logger.info(f"Analyzing {num_points} wavelength points")
        
        # Convert to meters for calculations
        lambda_0 = wavelengths_nm * 1e-9
        
        # Calculate effective index using polynomial fit - exact match to original
        # Original: neff = (n1 + n2*(lambda_0*1e6) + n3*(lambda_0*1e6)**2)
        neff = (self.neff_coeffs[0] + 
                self.neff_coeffs[1] * (lambda_0 * 1e6) + 
                self.neff_coeffs[2] * (lambda_0 * 1e6)**2)
        
        # Propagation constant
        beta0 = 2 * math.pi * neff / lambda_0
        
        # Convert loss from dB/cm to dB/m - exact match to original
        alpha = self.loss_db_per_cm * 100
        
        # Number of rings
        nor = len(self.ring_radii_um)
        logger.info(f"Simulating {nor} rings with radii: {self.ring_radii_um} μm")
        
        # Create R and phi arrays like the original script
        R = [radius * 1e-6 for radius in self.ring_radii_um]  # Convert μm to m
        R.append(0)  # Add dummy entry like original
        phi = list(self.phase_shifts_rad)
        phi.append(0)  # Add dummy entry like original
        
        # Calculate circumferences
        L = [radius * (2 * math.pi) for radius in R]
        
        # Initialize result arrays
        t1N, r1N, rN1, tN1 = [], [], [], []
        
        # Calculate response for each wavelength - exact match to original algorithm
        logger.info("Computing frequency response...")
        for i, beta in enumerate(beta0):
            if i % (len(beta0) // 10) == 0:
                progress = int(100 * i / len(beta0))
                logger.info(f"Progress: {progress}%")
                
            # Start with first ring - exact match to original
            S = self._calculate_ring_transfer_matrix(
                self.coupling_coeffs[0], phi[0], L[0], beta, alpha
            )
            M = self._scattering_to_transfer_matrix(np.array(S))
            
            # Cascade additional rings - exact match to original matrix multiplication order
            for no in range(nor):
                S_ring = self._calculate_ring_transfer_matrix(
                    self.coupling_coeffs[no + 1], phi[no + 1], L[no + 1], beta, alpha
                )
                Mtemp = self._scattering_to_transfer_matrix(np.array(S_ring))
                M = np.matmul(Mtemp, M)  # CRITICAL: same order as original script
            
            # Convert back to scattering parameters
            S_total = self._transfer_to_scattering_matrix(M)
            
            # Store results - exact match to original assignments
            t1N.append(S_total[0,0])  # Electric field Drop
            rN1.append(S_total[0,1])  # 
            r1N.append(S_total[1,0])  # Electric field Through
            tN1.append(S_total[1,1])  # 
        
        logger.info("System analysis complete")
        return wavelengths_nm, np.array(t1N), np.array(r1N), np.array(rN1), np.array(tN1)


class RingAnalyzer:
    """Utility class for analyzing ring resonator system responses."""
    
    @staticmethod
    def calculate_power_response(t1N: np.ndarray, r1N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate power transmission and reflection in dB.
        
        Args:
            t1N: Drop port field coefficients
            r1N: Through port field coefficients
            
        Returns:
            Tuple of (drop_power_dB, through_power_dB, total_power_dB)
        """
        drop_power = np.abs(t1N)**2
        through_power = np.abs(r1N)**2
        total_power = drop_power + through_power
        
        # Convert to dB (handle zeros by setting minimum value)
        min_val = 1e-12
        drop_dB = 10 * np.log10(np.maximum(drop_power, min_val))
        through_dB = 10 * np.log10(np.maximum(through_power, min_val))
        total_dB = 10 * np.log10(np.maximum(total_power, min_val))
        
        return drop_dB, through_dB, total_dB
    
    @staticmethod
    def calculate_phase_response(t1N: np.ndarray, r1N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate unwrapped phase response.
        
        Args:
            t1N: Drop port field coefficients
            r1N: Through port field coefficients
            
        Returns:
            Tuple of (drop_phase, through_phase) in radians
        """
        drop_phase = np.unwrap(np.angle(t1N))
        through_phase = np.unwrap(np.angle(r1N))
        
        return drop_phase, through_phase
    
    @staticmethod
    def calculate_group_delay(wavelengths: np.ndarray, through_phase: np.ndarray, 
                            drop_phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate group delay from phase response.
        
        Args:
            wavelengths: Wavelength array in meters
            through_phase: Through port phase in radians
            drop_phase: Drop port phase in radians
            
        Returns:
            Tuple of (through_delay, drop_delay) in seconds
        """
        c = 299792458  # speed of light in m/s
        frequencies = c / wavelengths
        omega = 2 * math.pi * frequencies
        
        through_delay = -np.diff(through_phase) / np.diff(omega)
        drop_delay = -np.diff(drop_phase) / np.diff(omega)
        
        return through_delay, drop_delay
    
    @staticmethod
    def calculate_dispersion(delays: Tuple[np.ndarray, np.ndarray], 
                           omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate group delay dispersion.
        
        Args:
            delays: Tuple of (through_delay, drop_delay)
            omega: Angular frequency array
            
        Returns:
            Tuple of (through_dispersion, drop_dispersion) in s²/rad
        """
        through_delay, drop_delay = delays
        omega_truncated = omega[:-1]  # Match delay array length
        
        through_dispersion = np.diff(through_delay) / np.diff(omega_truncated)
        drop_dispersion = np.diff(drop_delay) / np.diff(omega_truncated)
        
        return through_dispersion, drop_dispersion


class RingPlotter:
    """Utility class for plotting ring resonator system responses."""
    
    @staticmethod
    def _setup_professional_style():
        """Setup professional matplotlib styling."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 1.5,
            'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        })
    
    @staticmethod
    def plot_waveguide_effective_index(wavelength_start_nm: float, wavelength_stop_nm: float, 
                                     neff_coeffs: List[float],
                                     save_path: str = "waveguide_dispersion.pdf") -> None:
        """Plot waveguide effective index vs wavelength."""
        RingPlotter._setup_professional_style()
        
        # Generate wavelength array
        wavelengths_nm = np.linspace(wavelength_start_nm, wavelength_stop_nm, 1000)
        wl_um = wavelengths_nm * 1e-3  # convert nm to micrometers
        
        # Calculate effective index
        n_eff = (neff_coeffs[0] + 
                neff_coeffs[1] * wl_um + 
                neff_coeffs[2] * wl_um**2)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.plot(wavelengths_nm, n_eff, color='#1f77b4', linewidth=2)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Effective Index')
        ax.set_title('Waveguide Effective Index Dispersion')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_xlim(wavelengths_nm[0], wavelengths_nm[-1])
        
        # Add polynomial equation as text
        equation = f'$n_{{eff}} = {neff_coeffs[0]:.3f} + {neff_coeffs[1]:.3f}\\lambda + {neff_coeffs[2]:.3f}\\lambda^2$'
        ax.text(0.02, 0.98, equation, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_transmission_spectrum(wavelengths: np.ndarray, drop_dB: np.ndarray, 
                                 through_dB: np.ndarray, total_dB: np.ndarray,
                                 save_path: str = "rings_spectrum_log.pdf") -> None:
        """Plot transmission spectrum in dB scale."""
        RingPlotter._setup_professional_style()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # wavelengths are already in nm
        ax.plot(wavelengths, through_dB, label="Through Port", linewidth=1.5, alpha=0.9)
        ax.plot(wavelengths, drop_dB, label="Drop Port", linewidth=1.5, alpha=0.9)
        ax.plot(wavelengths, total_dB, label="Total Power", linewidth=1.5, alpha=0.9, linestyle='--')
        
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission (dB)")
        ax.set_title("Ring Resonator Transmission Spectrum")
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_phase_response(wavelengths: np.ndarray, through_phase: np.ndarray,
                          drop_phase: np.ndarray, save_path: str = "rings_phase.pdf") -> None:
        """Plot unwrapped phase response."""
        RingPlotter._setup_professional_style()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # wavelengths are already in nm
        ax.plot(wavelengths, through_phase / math.pi, label="Through Port Phase", 
                linewidth=1.5, alpha=0.9)
        ax.plot(wavelengths, drop_phase / math.pi, label="Drop Port Phase", 
                linewidth=1.5, alpha=0.9)
        
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Phase (π rad)")
        ax.set_title("Ring Resonator Phase Response")
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_group_delay(wavelengths: np.ndarray, through_delay: np.ndarray,
                        drop_delay: np.ndarray, save_path: str = "rings_delay.pdf") -> None:
        """Plot group delay response."""
        RingPlotter._setup_professional_style()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # wavelengths are already in nm, delay arrays are truncated by 1 point
        wl_nm = wavelengths[:-1]  # Truncated for delay calculation
        ax.plot(wl_nm, through_delay * 1e12, label="Through Port", 
                linewidth=1.5, alpha=0.9)
        ax.plot(wl_nm, drop_delay * 1e12, label="Drop Port", 
                linewidth=1.5, alpha=0.9)
        
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Group Delay (ps)")
        ax.set_title("Ring Resonator Group Delay")
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_xlim(wl_nm[0], wl_nm[-1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_dispersion(wavelengths: np.ndarray, through_dispersion: np.ndarray,
                       drop_dispersion: np.ndarray, save_path: str = "rings_dispersion.pdf") -> None:
        """Plot group delay dispersion."""
        RingPlotter._setup_professional_style()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # wavelengths are already in nm, dispersion arrays are truncated by 2 points
        wl_nm = wavelengths[:-2]  # Double truncated for dispersion calculation
        ax.plot(wl_nm, through_dispersion * 1e12, label="Through Port", 
                linewidth=1.5, alpha=0.9)
        ax.plot(wl_nm, drop_dispersion * 1e12, label="Drop Port", 
                linewidth=1.5, alpha=0.9)
        
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Group Delay Dispersion (ps²/nm)")
        ax.set_title("Ring Resonator Group Delay Dispersion")
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_xlim(wl_nm[0], wl_nm[-1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function demonstrating ring resonator system analysis.
    
    This example creates a 2-ring system and analyzes its complete response
    including transmission spectrum, phase, group delay, and dispersion.
    """
    # Initialize ring resonator system
    ring_system = RingResonatorSystem()
    
    # Configure the complete system in one call
    ring_system.configure(
        # Ring geometry (micrometers)
        ring_radii_um=[20.0, 15],
        
        # Coupling coefficients (power coupling)
        coupling_coeffs=[0.1, 0.005, 0.1],  # [bus-ring1, ring1-ring2]
        
        # Phase shifts (radians)
        phase_shifts_rad=[0.0, 0.0],
        
        # Wavelength range and resolution (nanometers)
        wavelength_start_nm=1520.0,
        wavelength_stop_nm=1570.0,
        wavelength_resolution_nm=0.005,
        
        # Material properties
        loss_db_per_cm=4.0,
        neff_coeffs=[4.077, -0.983, -0.046]  # [n1, n2, n3]
    )
    
    # Perform analysis
    print("\nAnalyzing ring resonator system...")
    wavelengths, t1N, r1N, rN1, tN1 = ring_system.analyze_system()
    
    # Calculate derived quantities
    analyzer = RingAnalyzer()
    drop_dB, through_dB, total_dB = analyzer.calculate_power_response(t1N, r1N)
    drop_phase, through_phase = analyzer.calculate_phase_response(t1N, r1N)
    
    c = 299792458  # speed of light in m/s
    wavelengths_m = wavelengths * 1e-9  # convert nm to m for calculations
    omega = 2 * math.pi * c / wavelengths_m
    through_delay, drop_delay = analyzer.calculate_group_delay(
        wavelengths_m, through_phase, drop_phase
    )
    through_dispersion, drop_dispersion = analyzer.calculate_dispersion(
        (through_delay, drop_delay), omega
    )
    
    # Generate plots
    print("Generating plots...")
    plotter = RingPlotter()
    
    # Plot waveguide effective index model
    plotter.plot_waveguide_effective_index(
        ring_system.wavelength_start, ring_system.wavelength_stop, 
        ring_system.neff_coeffs
    )
    
    # Plot ring resonator responses
    plotter.plot_transmission_spectrum(wavelengths, drop_dB, through_dB, total_dB)
    plotter.plot_phase_response(wavelengths, through_phase, drop_phase)
    plotter.plot_group_delay(wavelengths, through_delay, drop_delay)
    plotter.plot_dispersion(wavelengths, through_dispersion, drop_dispersion)
    
    print("Analysis complete. Plots saved as PDF files.")
    
    # Display final configuration
    print("\nFinal Configuration:")
    config = ring_system.get_configuration()
    for key, value in config.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

# Ring Resonator Design

A Python package for designing and simulating coupled ring resonator systems in silicon photonics. This package provides analysis capabilities for transmission spectra, phase response, group delay, and dispersion characteristics, with both command-line and graphical user interfaces.

## Features

- Multi-ring resonator systems (1 to N rings)
- Scattering matrix calculations with transfer matrix method
- Waveguide effective index dispersion modeling
- Modern graphical user interface with real-time visualization
- Interactive parameter control and configuration
- Export capabilities for plots and configurations

## Installation

```bash
pip install rings-resonator
```

Or install from source:

```bash
git clone https://github.com/SiEPIC/rings-design.git
cd rings-design
pip install -e .
```

## Usage

### Graphical User Interface

Launch the modern GUI application:

```bash
rings-gui
```

The GUI provides:
- Interactive parameter controls for rings, coupling, and wavelength settings
- Real-time ring geometry visualization
- Multi-tab interface with ring geometry and analysis plots
- Configuration save/load functionality
- Export capabilities for plots

### Command Line Interface

Run the command-line version:

```bash
rings-design
```

### Python API

```python
from rings_resonator import RingResonatorSystem, RingAnalyzer, RingPlotter

# Create and configure a 2-ring system
system = RingResonatorSystem()
system.configure(
    ring_radii_um=[35.0, 21.0],              # Ring radii in micrometers
    coupling_coeffs=[0.1, 0.001, 0.1],       # Power coupling coefficients
    wavelength_start_nm=1520.0,              # Start wavelength in nm
    wavelength_stop_nm=1570.0,               # Stop wavelength in nm
    wavelength_resolution_nm=0.005           # Resolution in nm
)

# Analyze the system
wavelengths, t1N, r1N, rN1, tN1 = system.analyze_system()

# Calculate power response
analyzer = RingAnalyzer()
drop_dB, through_dB, total_dB = analyzer.calculate_power_response(t1N, r1N)

# Generate plots
plotter = RingPlotter()
plotter.plot_transmission_spectrum(wavelengths, drop_dB, through_dB, total_dB)
```

## GUI Interface

The graphical interface includes:

### Ring Geometry Visualization
- Interactive display of ring resonator layout
- Real-time updates as parameters change
- Coupling coefficient annotations
- Waveguide port labeling

### Analysis Plots
- Transmission spectrum (through and drop ports)
- Phase response
- Group delay
- Group delay dispersion

### Parameter Controls
- Number of rings (1-5)
- Ring radii configuration
- Coupling coefficients
- Phase shifts
- Wavelength range and resolution
- Material properties

### Configuration Management
- Save/load configurations to JSON files
- Export plots to PDF
- Progress tracking for simulations

## Configuration Examples

**Single Ring:**
```python
system.configure(
    ring_radii_um=[50.0],
    coupling_coeffs=[0.05, 0.05],
    wavelength_start_nm=1500.0,
    wavelength_stop_nm=1600.0
)
```

**Multiple Rings with Phase Shifts:**
```python
system.configure(
    ring_radii_um=[40.0, 35.0, 30.0],
    coupling_coeffs=[0.08, 0.02, 0.02, 0.08],
    phase_shifts_rad=[0.0, 1.57, 3.14],
    wavelength_start_nm=1520.0,
    wavelength_stop_nm=1580.0
)
```

**Custom Material Properties:**
```python
system.configure(
    ring_radii_um=[30.0, 25.0],
    coupling_coeffs=[0.12, 0.008, 0.12],
    loss_db_per_cm=3.5,
    neff_coeffs=[4.1, -1.0, -0.05]
)
```

## Parameters

| Parameter | Description | Units | Example |
|-----------|-------------|--------|---------|
| `ring_radii_um` | Ring radii | micrometers | `[35.0, 21.0]` |
| `coupling_coeffs` | Power coupling coefficients | dimensionless | `[0.1, 0.001, 0.1]` |
| `phase_shifts_rad` | Phase shifts | radians | `[0.0, 1.57]` |
| `wavelength_start_nm` | Start wavelength | nanometers | `1520.0` |
| `wavelength_stop_nm` | Stop wavelength | nanometers | `1570.0` |
| `wavelength_resolution_nm` | Resolution | nanometers | `0.005` |
| `loss_db_per_cm` | Waveguide loss | dB/cm | `4.0` |
| `neff_coeffs` | Effective index coefficients | dimensionless | `[4.077, -0.983, -0.046]` |

## Common Wavelength Ranges

| Band | Range (nm) | Typical Resolution (nm) |
|------|------------|-------------------------|
| C-band | 1530-1565 | 0.01 |
| Extended C-band | 1520-1570 | 0.01 |
| C+L band | 1480-1625 | 0.02 |

## Generated Plots

The package generates five plots:

1. Waveguide Effective Index (`waveguide_dispersion.pdf`)
2. Transmission Spectrum (`rings_spectrum_log.pdf`)
3. Phase Response (`rings_phase.pdf`)
4. Group Delay (`rings_delay.pdf`)
5. Group Delay Dispersion (`rings_dispersion.pdf`)

## Implementation Details

The simulation uses scattering matrix calculations for coupled ring resonator systems:

- Transfer matrix method for cascaded analysis
- Polynomial effective index dispersion model: `n_eff = n₁ + n₂λ + n₃λ²`
- Propagation and coupling loss modeling
- Power conservation verification

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0
- CustomTkinter >= 5.0.0 (for GUI)

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research:

```bibtex
@software{rings_resonator,
  author = {Hammood, Mustafa},
  title = {Ring Resonator Design},
  url = {https://github.com/SiEPIC/rings-design},
  version = {1.0.0},
  year = {2024}
}
```

## Acknowledgments

Based on H. Shoman's MATLAB implementation. 
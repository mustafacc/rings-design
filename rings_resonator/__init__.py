"""
Ring Resonator Design Package

A Python package for designing and simulating coupled ring resonator systems
in silicon photonics.
"""

from .rings_design import RingResonatorSystem, RingAnalyzer, RingPlotter

__version__ = "1.0.0"
__author__ = "Mustafa Hammood"
__email__ = "mustafa@siepic.com"

__all__ = [
    "RingResonatorSystem",
    "RingAnalyzer", 
    "RingPlotter",
] 
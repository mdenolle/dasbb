"""
dasbb — Joint DAS + Broadband Bayesian Seismic Inversion
=========================================================

Fuses dense, low-aperture DAS fiber-optic arrays with sparse,
wide-aperture broadband networks for earthquake location and
velocity tomography, with proper heteroscedastic correlated
data covariance and full posterior uncertainty.
"""

from .data import DASPicks, BroadbandPicks, VelocityModel
from .forward import EikonalSolver
from .inversion import JointInversion, InversionConfig
from .design import OptimalDesign
from .information import InformationGain
from .weighting import AdaptiveWeighting
from .diagnostics import DesignDiagnostics, InversionDiagnostics
from .synthetic import (
    generate_synthetic_test,
    generate_alaska_scenario,
    generate_ocean_island_scenario,
)

__version__ = "0.1.0"

__all__ = [
    # Data
    "DASPicks", "BroadbandPicks", "VelocityModel",
    # Forward
    "EikonalSolver",
    # Inversion
    "JointInversion", "InversionConfig",
    # Design
    "OptimalDesign",
    # Information
    "InformationGain",
    # Weighting
    "AdaptiveWeighting",
    # Diagnostics
    "DesignDiagnostics", "InversionDiagnostics",
    # Synthetic
    "generate_synthetic_test",
    "generate_alaska_scenario",
    "generate_ocean_island_scenario",
]

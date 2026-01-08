"""
Data generation module for Feynman-Kac PINN.

Contains:
- Brownian motion simulation
- Domain definitions
- Boundary condition handlers
"""

from .brownian import simulate_brownian_paths, get_device
from .domains import Hypercube, Hypersphere, Domain

__all__ = [
    "simulate_brownian_paths",
    "get_device",
    "Hypercube",
    "Hypersphere",
    "Domain",
]

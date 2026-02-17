"""Problem definitions for Feynman-Kac PINN experiments."""

from .base import Problem
from .black_scholes import BlackScholesND, BlackScholesParams
from .schrodinger import HarmonicOscillatorND, HarmonicOscillatorParams

__all__ = [
    "Problem",
    "BlackScholesND",
    "BlackScholesParams",
    "HarmonicOscillatorND",
    "HarmonicOscillatorParams",
]

"""
Feynman-Kac PINN - Machine Learning Core

This module contains the neural network architectures, training logic,
and Monte Carlo random walk simulation for solving PDEs via the
Feynman-Kac formula.
"""

__version__ = "0.1.0"

from .problems import (
    BlackScholesND,
    HarmonicOscillatorND,
    Problem,
    available_problems,
    create_problem,
    default_problem_configs,
    deserialize_problem,
    serialize_problem,
)

__all__ = [
    "Problem",
    "BlackScholesND",
    "HarmonicOscillatorND",
    "available_problems",
    "default_problem_configs",
    "create_problem",
    "serialize_problem",
    "deserialize_problem",
]

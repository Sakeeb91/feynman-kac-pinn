"""
Data generation module for Feynman-Kac PINN.

Contains:
- Brownian motion simulation
- Domain definitions
- Boundary condition handlers
"""

from .brownian import (
    simulate_brownian_paths,
    simulate_brownian_paths_batched,
    feynman_kac_estimate,
    get_device,
)
from .domains import (
    Domain,
    Hypercube,
    Hypersphere,
    HyperEllipsoid,
    Interval,
    Union,
    Intersection,
    visualize_domain_samples,
)
from .boundary import (
    BoundaryCondition,
    FluxBoundaryCondition,
    DirichletBC,
    NeumannBC,
    sample_boundary_points,
    sample_boundary_values,
    estimate_outward_normals,
)

__all__ = [
    "simulate_brownian_paths",
    "simulate_brownian_paths_batched",
    "feynman_kac_estimate",
    "get_device",
    "Hypercube",
    "Hypersphere",
    "HyperEllipsoid",
    "Interval",
    "Union",
    "Intersection",
    "visualize_domain_samples",
    "BoundaryCondition",
    "FluxBoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "sample_boundary_points",
    "sample_boundary_values",
    "estimate_outward_normals",
    "Domain",
]

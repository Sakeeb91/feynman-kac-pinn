import math

import pytest

torch = pytest.importorskip("torch")

from ml.data.boundary import DirichletBC
from ml.data.brownian import (
    feynman_kac_estimate,
    get_device,
    profile_simulation_devices,
    simulate_brownian_paths,
)
from ml.data.domains import Hypersphere, Interval, visualize_domain_samples
from ml.utils.mc_estimator import FeynmanKacSolver


def test_get_device_returns_supported_backend() -> None:
    device = get_device()
    assert device in {"cpu", "mps", "cuda"}


def test_interval_contains_center_point() -> None:
    domain = Interval(0.0, 1.0)
    x = torch.tensor([[0.5]])
    assert domain.contains(x).item()


def test_exit_time_from_unit_sphere_matches_reference_value() -> None:
    domain = Hypersphere(center=0.0, radius=1.0, dim=3)
    x0 = torch.zeros(4096, 3)
    _, exit_times, _ = simulate_brownian_paths(
        x0=x0,
        dt=0.001,
        max_steps=20000,
        domain=domain,
        device="cpu",
        seed=7,
    )
    expected = 1.0 / 6.0
    relative_error = abs(exit_times.mean().item() - expected) / expected
    assert relative_error < 0.10


def test_heat_equation_value_at_midpoint() -> None:
    domain = Interval(0.0, 1.0)
    boundary_fn = lambda x: x.squeeze(-1)
    x = torch.tensor([[0.5]])
    estimates, _ = feynman_kac_estimate(
        x=x,
        boundary_fn=boundary_fn,
        domain=domain,
        n_paths=6000,
        dt=0.0005,
        max_steps=20000,
        device="cpu",
        antithetic=True,
    )
    rel_error = abs(estimates.item() - 0.5) / 0.5
    assert rel_error < 0.05

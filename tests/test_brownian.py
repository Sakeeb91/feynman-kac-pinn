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


def test_monte_carlo_variance_decays_like_inverse_sqrt_n() -> None:
    domain = Interval(0.0, 1.0)
    x = torch.tensor([[0.5]])
    boundary_fn = lambda y: y.squeeze(-1)

    sample_sizes = torch.tensor([500.0, 1000.0, 2000.0, 4000.0])
    std_errors = []
    for n in sample_sizes.tolist():
        _, std = feynman_kac_estimate(
            x=x,
            boundary_fn=boundary_fn,
            domain=domain,
            n_paths=int(n),
            dt=0.001,
            max_steps=15000,
            device="cpu",
            antithetic=True,
        )
        std_errors.append(std.item())

    x_log = torch.log(sample_sizes)
    y_log = torch.log(torch.tensor(std_errors))
    slope = ((x_log - x_log.mean()) * (y_log - y_log.mean())).sum() / ((x_log - x_log.mean()) ** 2).sum()
    intercept = y_log.mean() - slope * x_log.mean()
    fit = slope * x_log + intercept
    residual_ss = ((y_log - fit) ** 2).sum()
    total_ss = ((y_log - y_log.mean()) ** 2).sum().clamp_min(1e-12)
    r_squared = 1.0 - residual_ss / total_ss

    assert -0.65 < slope.item() < -0.35
    assert r_squared.item() > 0.95

"""
Brownian motion simulation for Feynman-Kac formula.

This module implements the core random walk engine that generates
Monte Carlo samples for PDE solution estimation.

The Feynman-Kac formula states that for suitable PDEs:
    u(x) = E[g(B_τ) · exp(-∫₀^τ c(B_s)ds)]

where:
    - B is Brownian motion starting at x
    - τ is the exit time from the domain
    - g is the boundary condition
    - c is the potential/reaction term
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional, Tuple

import torch

from .domains import Domain


ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True)
class SimulationOptions:
    """Tunable options for Brownian path simulation."""

    antithetic: bool = False
    stratified: bool = False
    batch_size: Optional[int] = None
    progress_callback: Optional[ProgressCallback] = None
    seed: Optional[int] = None


def _validate_inputs(x0: torch.Tensor, dt: float, max_steps: int, domain: Domain) -> None:
    if x0.dim() != 2:
        raise ValueError(f"x0 must be 2D (batch_size, dim), got shape {x0.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if x0.shape[1] != domain.dim:
        raise ValueError(f"x0 dimension ({x0.shape[1]}) doesn't match domain ({domain.dim})")
    if x0.shape[0] <= 0:
        raise ValueError("x0 must contain at least one path")


def get_device() -> str:
    """
    Auto-detect the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _stratified_normal(
    n_samples: int,
    dim: int,
    device: str,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw approximately stratified N(0,1) samples via inverse-CDF sampling."""
    u = (torch.arange(n_samples, device=device, dtype=dtype).unsqueeze(-1) + torch.rand(
        n_samples, dim, device=device, dtype=dtype, generator=generator
    )) / n_samples
    perm = torch.stack([torch.randperm(n_samples, device=device, generator=generator) for _ in range(dim)], dim=1)
    u = u[perm, torch.arange(dim, device=device)]
    z = torch.erfinv(2 * u - 1) * (2.0 ** 0.5)
    return z


def _sample_normal_increments(
    n_active: int,
    dim: int,
    sqrt_dt: float,
    device: str,
    dtype: torch.dtype,
    stratified: bool,
    antithetic: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if antithetic:
        half = (n_active + 1) // 2
        if stratified:
            base = _stratified_normal(half, dim, device, dtype, generator)
        else:
            base = torch.randn(half, dim, device=device, dtype=dtype, generator=generator)
        increments = torch.cat([base, -base], dim=0)[:n_active]
    elif stratified:
        increments = _stratified_normal(n_active, dim, device, dtype, generator)
    else:
        increments = torch.randn(n_active, dim, device=device, dtype=dtype, generator=generator)
    return increments * sqrt_dt


def simulate_brownian_paths(
    x0: torch.Tensor,
    dt: float,
    max_steps: int,
    domain: Domain,
    potential_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    antithetic: bool = False,
    stratified: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate Brownian motion paths until domain exit.

    Uses Euler-Maruyama discretization:
        X_{n+1} = X_n + sqrt(dt) * Z_n
    where Z_n ~ N(0, I).

    Args:
        x0: Starting points, shape (batch_size, dim)
        dt: Time step for discretization (smaller = more accurate, slower)
        max_steps: Maximum number of steps before timeout
        domain: Domain object defining the region and exit conditions
        potential_fn: Optional function c(x) for path integral computation.
                     If None, path_integrals will be zeros.
        device: Compute device ('mps', 'cuda', 'cpu'). Auto-detected if None.
        seed: Random seed for reproducibility. If None, uses current RNG state.
        antithetic: If True, use antithetic increments for variance reduction.
        stratified: If True, use stratified Gaussian draws per step.
        progress_callback: Optional callback `(step, max_steps, active_paths)`.

    Returns:
        exit_points: Points where paths exited, shape (batch_size, dim)
        exit_times: Time of exit for each path, shape (batch_size,)
        path_integrals: ∫₀^τ c(B_s)ds for each path, shape (batch_size,)

    Example:
        >>> domain = Hypercube(low=0.0, high=1.0, dim=2)
        >>> x0 = torch.full((100, 2), 0.5)  # Start in center
        >>> exit_pts, exit_times, integrals = simulate_brownian_paths(
        ...     x0, dt=0.001, max_steps=10000, domain=domain
        ... )
        >>> print(f"Mean exit time: {exit_times.mean():.4f}")
    """
    if device is None:
        device = get_device()
    _validate_inputs(x0=x0, dt=dt, max_steps=max_steps, domain=domain)
    batch_size, dim = x0.shape
    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Move to device
    x0 = x0.to(device)

    # Initialize tracking tensors
    positions = x0.clone()
    exit_points = torch.zeros_like(x0)
    exit_times = torch.zeros(batch_size, device=device)
    path_integrals = torch.zeros(batch_size, device=device)

    # Track which paths have exited
    active = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Verify starting points are inside domain
    if not domain.contains(x0).all():
        raise ValueError("Some starting points are outside the domain")

    sqrt_dt = dt ** 0.5
    current_time = 0.0

    for step in range(max_steps):
        n_active = active.sum().item()
        if n_active == 0:
            break

        current_time = (step + 1) * dt

        # Generate Brownian increments only for active paths
        dW = _sample_normal_increments(
            n_active=n_active,
            dim=dim,
            sqrt_dt=sqrt_dt,
            device=device,
            dtype=x0.dtype,
            stratified=stratified,
            antithetic=antithetic,
            generator=generator,
        )

        # Store previous positions for interpolation
        prev_positions = positions[active].clone()

        # Update positions (Euler-Maruyama step)
        positions[active] = positions[active] + dW

        # Accumulate path integral if potential provided
        # Use midpoint rule for better accuracy: c((x_prev + x_new)/2) * dt
        if potential_fn is not None:
            midpoints = (prev_positions + positions[active]) / 2
            path_integrals[active] = path_integrals[active] + potential_fn(midpoints) * dt

        # Check for exits
        inside = domain.contains(positions)
        newly_exited = active & ~inside

        if newly_exited.any():
            # Project to boundary (simple approach - clamp to domain)
            # For more accuracy, could interpolate exact crossing point
            exit_points[newly_exited] = domain.project_to_boundary(
                positions[newly_exited]
            )
            exit_times[newly_exited] = current_time
            active[newly_exited] = False
        if progress_callback is not None:
            progress_callback(step + 1, max_steps, n_active)

    # Handle paths that didn't exit (timeout)
    if active.any():
        # These paths are still inside - project current position to boundary
        # and mark as timed out
        exit_points[active] = domain.project_to_boundary(positions[active])
        exit_times[active] = max_steps * dt

    return exit_points, exit_times, path_integrals


def simulate_brownian_paths_batched(
    x0: torch.Tensor,
    dt: float,
    max_steps: int,
    domain: Domain,
    potential_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    antithetic: bool = False,
    stratified: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate Brownian paths in mini-batches to reduce peak memory usage.
    """
    if batch_size is None or batch_size >= x0.shape[0]:
        return simulate_brownian_paths(
            x0=x0,
            dt=dt,
            max_steps=max_steps,
            domain=domain,
            potential_fn=potential_fn,
            device=device,
            seed=seed,
            antithetic=antithetic,
            stratified=stratified,
            progress_callback=progress_callback,
        )

    n = x0.shape[0]
    all_exit_points: list[torch.Tensor] = []
    all_exit_times: list[torch.Tensor] = []
    all_integrals: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_seed = None if seed is None else seed + start
        result = simulate_brownian_paths(
            x0=x0[start:end],
            dt=dt,
            max_steps=max_steps,
            domain=domain,
            potential_fn=potential_fn,
            device=device,
            seed=batch_seed,
            antithetic=antithetic,
            stratified=stratified,
            progress_callback=progress_callback,
        )
        all_exit_points.append(result[0])
        all_exit_times.append(result[1])
        all_integrals.append(result[2])
    return (
        torch.cat(all_exit_points, dim=0),
        torch.cat(all_exit_times, dim=0),
        torch.cat(all_integrals, dim=0),
    )


def _synchronize_device(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def profile_simulation_devices(
    x0: torch.Tensor,
    dt: float,
    max_steps: int,
    domain: Domain,
    potential_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    include_cpu: bool = True,
) -> dict[str, float]:
    """
    Benchmark simulation runtime across available devices.

    Returns a mapping of device name to elapsed wall-clock seconds.
    """
    devices: list[str] = []
    if include_cpu:
        devices.append("cpu")
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    timings: dict[str, float] = {}
    for device in devices:
        start = perf_counter()
        simulate_brownian_paths_batched(
            x0=x0.to(device),
            dt=dt,
            max_steps=max_steps,
            domain=domain,
            potential_fn=potential_fn,
            device=device,
            batch_size=min(2048, x0.shape[0]),
            seed=0,
        )
        _synchronize_device(device)
        timings[device] = perf_counter() - start
    return timings


def feynman_kac_estimate(
    x: torch.Tensor,
    boundary_fn: Callable[[torch.Tensor], torch.Tensor],
    domain: Domain,
    potential_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    n_paths: int = 1000,
    dt: float = 0.001,
    max_steps: int = 10000,
    device: Optional[str] = None,
    antithetic: bool = False,
    stratified: bool = False,
    batch_size: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Feynman-Kac Monte Carlo estimate of PDE solution.

    For the PDE with solution u(x), computes:
        u(x) ≈ (1/M) Σ g(B_τ^j) · exp(-∫₀^τ c(B_s)ds)

    where M is the number of Monte Carlo paths.

    Args:
        x: Query points, shape (n_points, dim)
        boundary_fn: Boundary condition g(x) that returns values at boundary
        domain: Domain defining the PDE region
        potential_fn: Potential term c(x). If None, assumes c(x) = 0
        n_paths: Number of MC paths per query point
        dt: Time step for Brownian motion
        max_steps: Maximum steps before timeout
        device: Compute device
        antithetic: Use antithetic variates in path generation.
        stratified: Use stratified Gaussian increments.
        batch_size: Optional path mini-batch size for memory control.
        progress_callback: Optional callback `(step, max_steps, active_paths)`.

    Returns:
        estimates: MC estimates of u(x), shape (n_points,)
        std_errors: Standard errors of estimates, shape (n_points,)

    Example:
        >>> domain = Interval(0.0, 1.0)
        >>> # Heat equation with u(0)=0, u(1)=1
        >>> boundary_fn = lambda x: x.squeeze(-1)  # g(x) = x
        >>> x = torch.tensor([[0.5]])
        >>> estimate, std = feynman_kac_estimate(x, boundary_fn, domain)
        >>> # Should be close to 0.5 (solution at midpoint is mean of boundary)
    """
    if device is None:
        device = get_device()

    x = x.to(device)
    n_points, dim = x.shape

    estimates = torch.zeros(n_points, device=device)
    variances = torch.zeros(n_points, device=device)

    for i in range(n_points):
        # Replicate starting point for all paths
        x0 = x[i:i+1].expand(n_paths, -1).clone()

        # Simulate paths
        def _point_progress(step: int, total_steps: int, active: int) -> None:
            if progress_callback is None:
                return
            global_step = i * max_steps + step
            progress_callback(global_step, n_points * max_steps, active)

        exit_points, _, path_integrals = simulate_brownian_paths_batched(
            x0=x0,
            dt=dt,
            max_steps=max_steps,
            domain=domain,
            potential_fn=potential_fn,
            device=device,
            batch_size=batch_size,
            antithetic=antithetic,
            stratified=stratified,
            progress_callback=_point_progress,
        )

        # Compute Feynman-Kac functional
        # F = g(B_τ) * exp(-∫c ds)
        boundary_values = boundary_fn(exit_points)
        if potential_fn is not None:
            weights = torch.exp(-path_integrals)
            functional = boundary_values * weights
        else:
            functional = boundary_values

        # MC estimate and variance
        estimates[i] = functional.mean()
        variances[i] = functional.var() / n_paths  # Variance of mean estimator

    std_errors = torch.sqrt(variances)
    return estimates, std_errors


# Quick verification when run directly
if __name__ == "__main__":
    from .domains import Hypercube, Hypersphere, Interval

    print(f"Using device: {get_device()}")
    print()

    # Test 1: Exit time from unit square
    print("Test 1: Brownian exit from unit square")
    domain = Hypercube(low=0.0, high=1.0, dim=2)
    x0 = torch.full((1000, 2), 0.5)  # Start in center

    exit_pts, exit_times, _ = simulate_brownian_paths(
        x0, dt=0.001, max_steps=10000, domain=domain, seed=42
    )

    print(f"  Mean exit time: {exit_times.mean():.4f}")
    print(f"  Std exit time: {exit_times.std():.4f}")
    print(f"  Exit points shape: {exit_pts.shape}")
    print()

    # Test 2: Feynman-Kac for 1D heat equation
    print("Test 2: 1D Heat equation (Feynman-Kac)")
    domain_1d = Interval(0.0, 1.0)
    # Boundary condition: g(0) = 0, g(1) = 1 -> g(x) = x
    boundary_fn = lambda x: x.squeeze(-1)

    x_test = torch.tensor([[0.25], [0.5], [0.75]])
    estimates, std_errors = feynman_kac_estimate(
        x_test, boundary_fn, domain_1d, n_paths=5000, dt=0.0001
    )

    print("  x | Estimate | Expected | Std Error")
    print("  --|----------|----------|----------")
    for i, xi in enumerate(x_test):
        print(f"  {xi.item():.2f} | {estimates[i].item():.4f}   | {xi.item():.4f}    | {std_errors[i].item():.4f}")
    print()

    # Test 3: Exit time from unit sphere (has known analytical result)
    print("Test 3: Brownian exit from unit sphere (3D)")
    sphere = Hypersphere(center=0.0, radius=1.0, dim=3)
    x0_sphere = torch.zeros(1000, 3)  # Start at origin

    _, exit_times_sphere, _ = simulate_brownian_paths(
        x0_sphere, dt=0.001, max_steps=10000, domain=sphere, seed=42
    )

    # Theoretical: E[τ] = R²/(2d) = 1/(2*3) = 1/6 ≈ 0.167 for radius=1, dim=3
    expected_mean = 1.0 / 6
    actual_mean = exit_times_sphere.mean().item()
    print(f"  Mean exit time: {actual_mean:.4f}")
    print(f"  Expected (theory): {expected_mean:.4f}")
    print(f"  Relative error: {abs(actual_mean - expected_mean) / expected_mean * 100:.1f}%")

# Phase 3: Problem Definitions (Black-Scholes + Schrodinger)

## Black-Scholes Mapping

For a basket option under risk-neutral dynamics:

`dS_i(t) = r S_i(t) dt + sigma_i S_i(t) dW_i(t)`

with correlated Brownian motions, the pricing PDE is:

`dV/dt + (1/2) sum_ij sigma_i sigma_j rho_ij S_i S_j d2V/dS_i dS_j + r sum_i S_i dV/dS_i - rV = 0`.

Using Feynman-Kac, the option value is the discounted expectation of terminal payoff.

### Implementation Notes

- `BlackScholesND` stores problem parameters and a bounded log-price domain.
- Boundary condition computes basket call/put payoff from `S = S0 * exp(x)`.
- Potential is constant `c(x) = r`.
- 1D case includes closed-form Black-Scholes analytical formula for validation.
- Correlation utilities return both matrix and Cholesky factor for simulation workflows.

## Schrodinger Harmonic Oscillator Mapping

For the `N`-dimensional harmonic oscillator:

`H psi = [-(hbar^2 / 2m) Delta + (1/2) m omega^2 |x|^2] psi = E psi`.

The ground state is Gaussian:

`psi_0(x) = C exp(-alpha |x|^2 / 2)`, where `alpha = m omega / hbar`.

Ground-state energy:

`E_0 = (N/2) hbar omega`.

### Implementation Notes

- `HarmonicOscillatorND` uses a hypersphere domain for radial decay.
- Potential is `V(x) = 0.5 * m * omega^2 * |x|^2`.
- Boundary condition and analytical solution both use the Gaussian ground state.
- Class exposes `alpha`, `ground_state_energy`, and expected boundary magnitude.

## Registry and API Integration

Phase 3 introduces `ml.problems` registry helpers:

- `available_problems()`
- `default_problem_configs()`
- `create_problem(name, **params)`
- `serialize_problem(problem)` / `deserialize_problem(payload)`

This enables clean problem selection and parameter round-tripping for backend API layers.

## Verification

Tests in `tests/test_problems.py` cover:

1. 1D Black-Scholes reference prices (call/put) within 1%.
2. Harmonic oscillator Gaussian behavior and positive energy/potential.
3. Registry defaults, serialization round-trip, and trainer adapter integration.

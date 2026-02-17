# Phase 1: Random Walk Engine Derivation

## PDE to Expectation via Feynman-Kac

Consider the elliptic boundary value problem on domain `Omega`:

`(1/2) Delta u(x) - c(x) u(x) = 0`, for `x in Omega`

with boundary condition:

`u(x) = g(x)`, for `x in partial Omega`.

Let `B_t` be Brownian motion started at `x`, and define the first exit time:

`tau = inf { t >= 0 : B_t notin Omega }`.

The Feynman-Kac representation is:

`u(x) = E[ g(B_tau) exp( - integral_0^tau c(B_s) ds ) ]`.

## Discretization Used in Code

The implementation uses Euler-Maruyama:

`X_{k+1} = X_k + sqrt(dt) * Z_k`, where `Z_k ~ N(0, I)`.

A single Monte Carlo sample contributes:

`F_j = g(X_tau_j) * exp(-I_j)`,

`I_j = sum_k c((X_k + X_{k+1}) / 2) * dt`.

The midpoint rule for `I_j` improves integral accuracy versus left-point quadrature.

## Monte Carlo Estimator

For `N` paths from the same starting point:

`u_hat_N(x) = (1/N) sum_{j=1}^N F_j`.

Standard error is estimated by:

`se(u_hat_N) = sqrt( Var(F) / N )`.

The implementation also supports:

- Antithetic variates: pair `Z` and `-Z` increments.
- Stratified Gaussian sampling: inverse-CDF stratification per step.
- Batching: split path simulation into smaller chunks to cap memory use.

## Verification Targets

Phase 1 tests verify:

1. Sphere exit-time reference check in 3D.
2. 1D heat-equation benchmark at `x=0.5`.
3. Empirical `O(1/sqrt(N))` Monte Carlo error decay.

## Practical Notes

- Device auto-selection prefers MPS, then CUDA, then CPU.
- `profile_simulation_devices(...)` benchmarks runtime across available devices.
- `FeynmanKacSolver` adds confidence intervals and adaptive sample sizing.

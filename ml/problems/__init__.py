"""Problem definitions for Feynman-Kac PINN experiments."""

from .base import Problem
from .black_scholes import BlackScholesND, BlackScholesParams
from .schrodinger import HarmonicOscillatorND, HarmonicOscillatorParams


_REGISTRY: dict[str, type[Problem]] = {
    "black_scholes": BlackScholesND,
    "harmonic_oscillator": HarmonicOscillatorND,
}


def available_problems() -> tuple[str, ...]:
    """List registered problem keys."""
    return tuple(sorted(_REGISTRY))


def default_problem_configs() -> dict[str, dict[str, object]]:
    """Return default parameter sets for each registered problem."""
    return {
        "black_scholes": BlackScholesParams().__dict__.copy(),
        "harmonic_oscillator": HarmonicOscillatorParams().__dict__.copy(),
    }


def create_problem(problem_name: str, **params: object) -> Problem:
    """Instantiate a problem from registry key and kwargs."""
    key = problem_name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown problem '{problem_name}'. choices: {', '.join(available_problems())}")
    return _REGISTRY[key](**params)


__all__ = [
    "Problem",
    "BlackScholesND",
    "BlackScholesParams",
    "HarmonicOscillatorND",
    "HarmonicOscillatorParams",
    "available_problems",
    "default_problem_configs",
    "create_problem",
]

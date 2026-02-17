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


def serialize_problem(problem: Problem) -> dict[str, object]:
    """Serialize an instantiated problem into a JSON-safe payload."""
    payload = problem.to_dict()
    for key, cls in _REGISTRY.items():
        if isinstance(problem, cls):
            payload["problem_name"] = key
            break
    return payload


def deserialize_problem(payload: dict[str, object]) -> Problem:
    """Recreate a problem from `serialize_problem` output."""
    if "problem_name" not in payload:
        raise ValueError("payload missing 'problem_name'")
    key = str(payload["problem_name"])
    params = payload.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("'parameters' must be a dictionary")
    return create_problem(key, **params)


__all__ = [
    "Problem",
    "BlackScholesND",
    "BlackScholesParams",
    "HarmonicOscillatorND",
    "HarmonicOscillatorParams",
    "available_problems",
    "default_problem_configs",
    "create_problem",
    "serialize_problem",
    "deserialize_problem",
]

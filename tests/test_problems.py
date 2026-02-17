import pytest

torch = pytest.importorskip("torch")

from ml.problems import BlackScholesND, HarmonicOscillatorND


def test_black_scholes_problem_instantiates() -> None:
    problem = BlackScholesND(dim=10)
    assert problem.dimension == 10
    assert problem.domain.dim == 10


def test_harmonic_oscillator_problem_instantiates() -> None:
    problem = HarmonicOscillatorND(dim=8)
    assert problem.dimension == 8
    assert problem.domain.dim == 8

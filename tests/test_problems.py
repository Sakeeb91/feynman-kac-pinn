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


def test_problem_metadata_contains_required_fields() -> None:
    bs = BlackScholesND(dim=2)
    meta = bs.metadata
    assert meta["name"] == "2D Black-Scholes"
    assert meta["dimension"] == 2
    assert "parameters" in meta
    serialized = bs.to_dict()
    assert serialized["class_name"] == "BlackScholesND"


def test_black_scholes_1d_matches_known_reference_prices() -> None:
    x = torch.tensor([[0.0]], dtype=torch.float64)  # S = S0
    call = BlackScholesND(
        dim=1,
        strike=100.0,
        initial_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        maturity=1.0,
        option_type="call",
    )
    put = BlackScholesND(
        dim=1,
        strike=100.0,
        initial_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        maturity=1.0,
        option_type="put",
    )
    call_price = call.analytical_solution(x).item()
    put_price = put.analytical_solution(x).item()
    assert call_price == pytest.approx(10.4506, rel=0.01)
    assert put_price == pytest.approx(5.5735, rel=0.01)


def test_black_scholes_boundary_and_correlation_shape() -> None:
    problem = BlackScholesND(dim=5, option_type="call", correlation=0.25)
    x = torch.zeros(7, 5)
    payoff = problem.boundary_condition(x)
    potential = problem.potential(x)
    corr = problem.correlation_matrix()
    chol = problem.cholesky_correlation()
    assert payoff.shape == (7,)
    assert potential.shape == (7,)
    assert corr.shape == (5, 5)
    assert chol.shape == (5, 5)
    assert torch.allclose(torch.diag(corr), torch.ones(5))

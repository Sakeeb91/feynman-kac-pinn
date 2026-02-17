import pytest

torch = pytest.importorskip("torch")

from ml.problems import (
    BlackScholesND,
    HarmonicOscillatorND,
    available_problems,
    create_problem,
    default_problem_configs,
    deserialize_problem,
    serialize_problem,
)
from ml.training.trainer import FKProblem


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


def test_harmonic_oscillator_ground_state_is_gaussian() -> None:
    problem = HarmonicOscillatorND(dim=3, mass=1.0, omega=1.0, hbar=1.0)
    x = torch.randn(128, 3)
    psi = problem.analytical_solution(x)
    boundary = problem.boundary_condition(x)
    relative_l2 = torch.norm(psi - boundary) / torch.norm(psi).clamp_min(1e-12)
    assert relative_l2.item() < 0.05


def test_harmonic_oscillator_energy_and_potential_are_positive() -> None:
    problem = HarmonicOscillatorND(dim=4, mass=2.0, omega=0.5, hbar=1.0)
    x = torch.randn(16, 4)
    potential = problem.potential(x)
    assert problem.ground_state_energy > 0.0
    assert torch.all(potential >= 0)


def test_problem_registry_lists_expected_keys() -> None:
    keys = available_problems()
    assert "black_scholes" in keys
    assert "harmonic_oscillator" in keys


def test_problem_serialization_round_trip() -> None:
    original = BlackScholesND(dim=3, strike=105.0, option_type="put")
    payload = serialize_problem(original)
    restored = deserialize_problem(payload)
    assert isinstance(restored, BlackScholesND)
    assert restored.get_parameters() == original.get_parameters()


def test_default_configs_create_problems() -> None:
    configs = default_problem_configs()
    for key, params in configs.items():
        problem = create_problem(key, **params)
        assert problem.dimension > 0


def test_fkproblem_adapter_from_problem() -> None:
    problem = HarmonicOscillatorND(dim=2)
    fk_problem = FKProblem.from_problem(problem)
    x = torch.randn(8, 2)
    boundary = fk_problem.boundary_condition(x)
    potential = fk_problem.potential(x) if fk_problem.potential is not None else None
    assert boundary.shape == (8,)
    assert potential is not None and potential.shape == (8,)

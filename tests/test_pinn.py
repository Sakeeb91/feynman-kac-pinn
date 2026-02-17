import pytest

torch = pytest.importorskip("torch")

from ml.models.pinn import FeynmanKacPINN
from ml.models.activations import available_activations, get_activation, swish
from ml.models.initialization import init_linear_layer
from ml.training.trainer import FKProblem, FeynmanKacTrainer


def test_forward_output_shape() -> None:
    model = FeynmanKacPINN(input_dim=3, hidden_dims=(16, 16), activation="tanh")
    x = torch.randn(8, 3)
    y = model(x)
    assert y.shape == (8, 1)


def test_invalid_shape_raises() -> None:
    model = FeynmanKacPINN(input_dim=2, hidden_dims=(8, 8), activation="gelu")
    with pytest.raises(ValueError):
        model(torch.randn(8, 3))


def test_activation_registry_contains_required_options() -> None:
    names = available_activations()
    for key in ("tanh", "swish", "gelu"):
        assert key in names
    assert get_activation("swish").__class__.__name__ == "SiLU"
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    out = swish(x)
    assert out.shape == x.shape


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError):
        get_activation("unknown")


def test_initialization_sets_zero_bias() -> None:
    layer = torch.nn.Linear(4, 3)
    init_linear_layer(layer, mode="xavier")
    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))


def test_he_initialization_runs() -> None:
    layer = torch.nn.Linear(4, 3)
    init_linear_layer(layer, mode="he")
    assert layer.weight.std().item() > 0.0


class _FixedPointDomain:
    dim = 1

    def __init__(self, point: float):
        self.point = float(point)

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        return torch.full((n, 1), self.point, device=device)


def _constant_target_estimator(
    x: torch.Tensor,
    problem: FKProblem,
    n_mc_paths: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    del problem, n_mc_paths, device
    targets = torch.full((x.shape[0],), 0.75, dtype=x.dtype, device=x.device)
    std = torch.zeros_like(targets)
    return targets, std


def test_trainer_can_overfit_single_point() -> None:
    domain = _FixedPointDomain(0.5)
    model = FeynmanKacPINN(input_dim=1, hidden_dims=(32, 32), activation="tanh")
    trainer = FeynmanKacTrainer(
        model=model,
        problem=FKProblem(domain=domain, boundary_condition=lambda x: x.squeeze(-1)),
        device="cpu",
        lr=3e-2,
        max_grad_norm=5.0,
        target_estimator=_constant_target_estimator,
    )
    for _ in range(200):
        trainer.train_step(batch_size=16, n_mc_paths=32)
    x = domain.sample_interior(1, device="cpu")
    pred = model(x).squeeze(-1).item()
    assert abs(pred - 0.75) < 1e-2


def test_training_loss_decreases_over_short_run() -> None:
    domain = _FixedPointDomain(0.25)
    model = FeynmanKacPINN(input_dim=1, hidden_dims=(16, 16), activation="gelu")
    trainer = FeynmanKacTrainer(
        model=model,
        problem=FKProblem(domain=domain, boundary_condition=lambda x: x.squeeze(-1)),
        device="cpu",
        lr=1e-2,
        target_estimator=_constant_target_estimator,
    )
    history = trainer.fit(steps=60, batch_size=8, n_mc_paths=8)
    assert history.train_loss[-1] < history.train_loss[0]

import pytest

torch = pytest.importorskip("torch")

from ml.models.pinn import FeynmanKacPINN
from ml.models.activations import available_activations, get_activation, swish


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

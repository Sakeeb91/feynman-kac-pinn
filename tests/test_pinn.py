import pytest

torch = pytest.importorskip("torch")

from ml.models.pinn import FeynmanKacPINN


def test_forward_output_shape() -> None:
    model = FeynmanKacPINN(input_dim=3, hidden_dims=(16, 16), activation="tanh")
    x = torch.randn(8, 3)
    y = model(x)
    assert y.shape == (8, 1)


def test_invalid_shape_raises() -> None:
    model = FeynmanKacPINN(input_dim=2, hidden_dims=(8, 8), activation="gelu")
    with pytest.raises(ValueError):
        model(torch.randn(8, 3))

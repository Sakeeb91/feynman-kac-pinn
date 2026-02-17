import torch

from ml.data.brownian import get_device
from ml.data.domains import Interval


def test_get_device_returns_supported_backend() -> None:
    device = get_device()
    assert device in {"cpu", "mps", "cuda"}


def test_interval_contains_center_point() -> None:
    domain = Interval(0.0, 1.0)
    x = torch.tensor([[0.5]])
    assert domain.contains(x).item()

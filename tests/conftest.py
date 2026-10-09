import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def fake_loader():
    torch.manual_seed(0)
    images = torch.randn(16, 1, 8, 8)
    labels = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


@pytest.fixture
def tiny_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )

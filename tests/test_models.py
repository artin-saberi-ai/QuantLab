import pytest
import torch

from quantlab.data import dataset_spec
from quantlab.models import build_model


@pytest.mark.parametrize(
    "name,channels,size",
    [
        ("mnist", 1, 28),
        ("fashion_mnist", 1, 28),
        ("cifar10", 3, 32),
    ],
)
def test_dataset_spec_shape(name, channels, size):
    spec = dataset_spec(name)
    assert spec["channels"] == channels
    assert spec["size"] == size
    assert spec["num_classes"] == 10
    assert len(spec["mean"]) == channels


def test_dataset_spec_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown dataset"):
        dataset_spec("imagenet")


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("vgg16", in_channels=3, num_classes=10, image_size=32)


def test_simple_cnn_output_shape():
    model = build_model("simple_cnn", in_channels=1, num_classes=10, image_size=28)
    assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)


def test_simple_cnn_supports_other_image_sizes():
    model = build_model("simple_cnn", in_channels=1, num_classes=10, image_size=32, widths=[16, 32])
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 10)


def test_resnet20_output_shape():
    model = build_model("resnet20", in_channels=3, num_classes=10, image_size=32)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_500_000

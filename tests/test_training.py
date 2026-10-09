import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quantlab.evaluation import evaluate
from quantlab.training import make_optimizer, set_seed, train


def test_set_seed_makes_training_deterministic(fake_loader):
    def build():
        return nn.Sequential(nn.Flatten(), nn.Linear(64, 10))

    set_seed(7)
    first = train(build(), fake_loader, fake_loader, epochs=1, lr=1e-2, log_every=0)

    set_seed(7)
    second = train(build(), fake_loader, fake_loader, epochs=1, lr=1e-2, log_every=0)

    assert first["train_loss"] == pytest.approx(second["train_loss"])
    assert first["test_accuracy"] == pytest.approx(second["test_accuracy"])


def test_train_single_epoch_records_history(tiny_cnn, fake_loader):
    history = train(tiny_cnn, fake_loader, fake_loader, epochs=1, lr=1e-2, log_every=0)

    assert set(history) == {"train_loss", "train_accuracy", "test_accuracy"}
    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0
    assert 0.0 <= history["test_accuracy"][0] <= 1.0


def test_train_rejects_unknown_optimizer(tiny_cnn):
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer(tiny_cnn, "lion", lr=1e-3)


def test_optimizer_selection(tiny_cnn):
    assert isinstance(make_optimizer(tiny_cnn, "adam", 1e-3), torch.optim.Adam)
    assert isinstance(make_optimizer(tiny_cnn, "sgd", 0.1), torch.optim.SGD)


class AlwaysClassZero(nn.Module):
    def forward(self, inputs):
        logits = torch.zeros(inputs.size(0), 10)
        logits[:, 0] = 100.0
        return logits


def test_evaluate_accuracy_on_deterministic_model():
    images = torch.randn(12, 1, 8, 8)
    labels = torch.zeros(12, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    result = evaluate(AlwaysClassZero(), loader)
    assert result["accuracy"] == 1.0
    assert result["samples"] == 12
    assert result["loss"] < 1.0


def test_evaluate_respects_max_batches():
    images = torch.randn(16, 1, 8, 8)
    labels = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    result = evaluate(AlwaysClassZero(), loader, max_batches=2)
    assert result["samples"] == 8

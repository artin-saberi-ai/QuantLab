import random

import torch
import torch.nn.functional as F

from .evaluation import evaluate


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, name, lr, weight_decay=0.0):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer '{name}' (expected adam, adamw or sgd)")


def train(
    model,
    train_loader,
    test_loader,
    *,
    epochs,
    lr,
    device="cpu",
    optimizer="adam",
    weight_decay=0.0,
    log_every=200,
):
    model.to(device)
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    history = {"train_loss": [], "train_accuracy": [], "test_accuracy": []}

    for epoch in range(epochs):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)

            if log_every and (step + 1) % log_every == 0:
                print(f"  epoch {epoch + 1} step {step + 1} loss {loss_sum / seen:.4f}")

        train_loss = loss_sum / max(seen, 1)
        train_acc = correct / max(seen, 1)
        test_acc = evaluate(model, test_loader, device)["accuracy"]
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"loss {train_loss:.4f} train_acc {train_acc:.4f} test_acc {test_acc:.4f}"
        )

    return history

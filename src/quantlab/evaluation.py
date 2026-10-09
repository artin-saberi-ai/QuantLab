import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device="cpu", max_batches=None):
    model.eval()
    model.to(device)
    param = next(model.parameters(), None)

    total, correct, loss_sum = 0, 0, 0.0
    for index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        if param is not None and param.dtype == torch.float16:
            inputs = inputs.half()
        logits = model(inputs)
        loss_sum += F.cross_entropy(logits.float(), targets, reduction="sum").item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    return {
        "accuracy": correct / total if total else 0.0,
        "loss": loss_sum / total if total else 0.0,
        "samples": total,
    }

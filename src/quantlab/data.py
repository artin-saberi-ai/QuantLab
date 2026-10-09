import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASET_SPECS = {
    "mnist": {
        "cls": datasets.MNIST,
        "channels": 1,
        "size": 28,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
        "train_transform": None,
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "channels": 1,
        "size": 28,
        "num_classes": 10,
        "mean": (0.2860,),
        "std": (0.3530,),
        "train_transform": None,
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "channels": 3,
        "size": 32,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "train_transform": "cifar",
    },
}


def dataset_spec(name):
    try:
        spec = DATASET_SPECS[name]
    except KeyError:
        raise ValueError(f"unknown dataset '{name}' (expected one of {sorted(DATASET_SPECS)})") from None
    return spec


def _transform(spec, train):
    ops = []
    if spec["train_transform"] == "cifar" and train:
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    return transforms.Compose(ops)


def make_loaders(name, root, batch_size, num_workers=0, seed=0):
    spec = dataset_spec(name)
    dataset_cls = spec["cls"]
    train_set = dataset_cls(root=root, train=True, download=True, transform=_transform(spec, train=True))
    test_set = dataset_cls(root=root, train=False, download=True, transform=_transform(spec, train=False))

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import QMNIST
from torchvision.transforms import ToTensor


def get_qmnist(transform=None, target_transform=None):
    train = QMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    test = QMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    return train, test


def collate_batch(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def get_qmnist_loader(transform=ToTensor(), target_transform=torch.tensor, **kwargs):
    train, test = get_qmnist(transform, target_transform)
    kwargs.setdefault("collate_fn", collate_batch)
    train_loader = DataLoader(train, **kwargs)
    test_kwargs = kwargs.copy()
    test_kwargs["shuffle"] = False
    test_loader = DataLoader(test, **test_kwargs)
    return train_loader, test_loader

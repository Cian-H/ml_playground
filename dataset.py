import torch
from torch.utils.data import DataLoader
from torchvision.datasets import QMNIST
from torchvision.transforms import ToTensor
import numpy as np
from pathlib import Path


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


def dl_to_numpy(dl):
    return map(np.asarray, zip(*dl))


def get_qmnist_numpy(**kwargs):
    data_path = Path("data/QMNIST/data.npz")
    if data_path.exists():
        data = np.load(data_path)
        train_X = data["train_X"]
        train_y = data["train_y"]
        test_X = data["test_X"]
        test_y = data["test_y"]
    else:
        train_dl, test_dl = get_qmnist_loader(**kwargs)
        (train_X, train_y), (test_X, test_y) = (
            dl_to_numpy(train_dl),
            dl_to_numpy(test_dl),
        )
        np.savez(
            data_path,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            test_y=test_y,
        )
    return ((train_X, train_y), (test_X, test_y))

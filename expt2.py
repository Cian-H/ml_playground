import torch
from tqdm.auto import tqdm, trange
import numpy as np

import config
from dataset import get_qmnist_loader
from results import display_results


__config = config.get()["torch"]


def run(
    lr: float = __config["lr"],
    momentum_decay: float = __config["momentum_decay"],
    batch_size: int = __config["batch_size"],
    n_epochs: int = __config["n_epochs"],
):
    train_loader, test_loader = get_qmnist_loader(batch_size=batch_size)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=6, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 10, kernel_size=6, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(160, 10),
        torch.nn.Softmax(dim=1),
    ).cuda()
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=lr, momentum_decay=momentum_decay
    )
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in trange(n_epochs, desc="Epoch"):
        model.train()
        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x = x.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.cuda()
                y_pred = model(x)
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted.cpu() == y).sum().item()

        tqdm.write(f"Epoch {epoch} Accuracy: {correct/total:.2f}")

    pred_y = np.concat(
        tuple(
            model(x[0].cuda()).argmax(axis=1).cpu().detach().numpy()
            for x in test_loader
        )
    )
    test_y = (
        torch.concat(tuple(x[1] for x in test_loader)).flatten().cpu().detach().numpy()
    )
    return display_results(test_y, pred_y)


if __name__ == "__main__":
    app = config.get()["app"]
    app.command()(run)
    app()

import torch
from typing import Optional
from tqdm.auto import tqdm, trange
import numpy as np

import config
from dataset import get_qmnist_loader
from results import display_results


__config = config.get()["torch"]
__app = config.get()["app"]


@__app.command()
def main(
    lr: Optional[float] = __config["lr"],
    momentum_decay: Optional[float] = __config["momentum_decay"],
    batch_size: Optional[int] = __config["batch_size"],
):
    run(lr=lr, momentum_decay=momentum_decay, batch_size=batch_size)


def run(
    lr: Optional[float] = __config["lr"],
    momentum_decay: Optional[float] = __config["momentum_decay"],
    batch_size: Optional[int] = __config["batch_size"],
):
    train_loader, test_loader = get_qmnist_loader(batch_size=batch_size)
    # A simple transformer model for solving the MNIST dataset
    img_size = 28 * 28
    encode_size = round(28 * 28, -1)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(img_size, encode_size),
        torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=encode_size, nhead=10),
            num_layers=1,
            enable_nested_tensor=False,
        ),
        torch.nn.Linear(encode_size, 10),
        torch.nn.Softmax(dim=1),
    ).cuda()
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=lr, momentum_decay=momentum_decay
    )
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in trange(5, desc="Epoch"):
        model.train()
        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x = x.cuda().view(-1, 28 * 28)
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
                x = x.cuda().view(-1, 28 * 28)
                y_pred = model(x)
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted.cpu() == y).sum().item()

        tqdm.write(f"Epoch {epoch} Accuracy: {correct/total:.2f}")

    pred_y = np.concat(
        tuple(
            model(x[0].cuda().view(-1, 28 * 28)).argmax(axis=1).cpu().detach().numpy()
            for x in test_loader
        )
    )
    test_y = (
        torch.concat(tuple(x[1] for x in test_loader)).flatten().cpu().detach().numpy()
    )
    display_results(test_y, pred_y)


if __name__ == "__main__":
    __app()

import torch
from typing import Optional
from tqdm.auto import tqdm, trange
import numpy as np

import config
from dataset import get_qmnist_loader
from results import display_results


__config = config.get()["torch"]
__app = config.get()["app"]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(
            28,
            10,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.pool_y = torch.nn.AvgPool2d(4, 2)
        self.pool_h = torch.nn.AvgPool2d(4, 1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(124, 10)
        self.out = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y, h = self.rnn(x)
        y = self.pool_y(y)
        y = self.flatten(y)
        h = h.transpose(0, 1)
        h = self.pool_h(h)
        h = self.flatten(h)
        yh = torch.cat((y, h), dim=1)
        yh = self.linear(yh)
        return self.out(yh)


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
    model = Model().cuda()
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=lr, momentum_decay=momentum_decay
    )
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in trange(5, desc="Epoch"):
        model.train()
        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x = x.cuda().view(-1, 28, 28)
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
                x = x.cuda().view(-1, 28, 28)
                y_pred = model(x)
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted.cpu() == y).sum().item()

        tqdm.write(f"Epoch {epoch} Accuracy: {correct/total:.2f}")

    pred_y = np.concat(
        tuple(
            model(x[0].cuda().view(-1, 28, 28)).argmax(axis=1).cpu().detach().numpy()
            for x in test_loader
        )
    )
    test_y = (
        torch.concat(tuple(x[1] for x in test_loader)).flatten().cpu().detach().numpy()
    )
    display_results(test_y, pred_y)


if __name__ == "__main__":
    __app()

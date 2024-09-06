import torch
from tqdm.auto import tqdm, trange
import numpy as np

from dataset import get_qmnist_loader
from results import display_results


def main():
    train_loader, test_loader = get_qmnist_loader()
    model = torch.nn.Sequential(
        torch.nn.Linear(28 * 28, 10),
        torch.nn.Softmax(dim=1),
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in trange(5):
        model.train()
        for x, y in tqdm(train_loader, desc="Training"):
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
            for x, y in tqdm(test_loader, desc="Testing"):
                x = x.cuda().view(-1, 28 * 28)
                y_pred = model(x)
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted.cpu() == y).sum().item()

        print(f"Accuracy: {correct/total:.2f}")

    pred_y = np.concat(
        tuple(
            model(x[0].cuda().view(-1, 28 * 28)).argmax(axis=1).cpu().detach().numpy()
            for x in test_loader
        )
    )
    test_y = (
        torch.stack(tuple(x[1] for x in test_loader)).flatten().cpu().detach().numpy()
    )
    display_results(test_y, pred_y)


if __name__ == "__main__":
    main()

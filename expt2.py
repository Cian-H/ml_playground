import torch
from tqdm.auto import tqdm, trange

from dataset import get_qmnist_loader


def main():
    train_loader, test_loader = get_qmnist_loader()
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 7 * 7, 10),
        torch.nn.Softmax(dim=1),
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in trange(5):
        model.train()
        for x, y in tqdm(train_loader, desc="Training"):
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
            for x, y in tqdm(test_loader, desc="Testing"):
                x = x.cuda()
                y_pred = model(x)
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted.cpu() == y).sum().item()

        print(f"Accuracy: {correct/total:.2f}")


if __name__ == "__main__":
    main()

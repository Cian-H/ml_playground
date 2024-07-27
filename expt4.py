import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

from dataset import get_qmnist


def main():
    train, test = get_qmnist()
    sc = StandardScaler()
    train_x = sc.fit_transform(train.data.view(-1, 28 * 28).cpu().numpy())
    train_y = train.targets.cpu().numpy().argmax(axis=1)
    train_xy = xgb.DMatrix(train_x, label=train_y)
    test_x = sc.fit_transform(test.data.view(-1, 28 * 28).cpu().numpy())
    test_y = test.targets.cpu().numpy().argmax(axis=1)
    test_xy = xgb.DMatrix(test_x, label=test_y)
    model = xgb.train(
        {
            "objective": "multi:softmax",
            "num_class": 10,
            "tree_method": "hist",
            "device": "cuda",
        },
        train_xy,
        num_boost_round=10,
    )
    accuracy = np.mean(model.predict(test_xy) == test_y)
    print(f"Accuracy: {accuracy:.8f}")


if __name__ == "__main__":
    main()

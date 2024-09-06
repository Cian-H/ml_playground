import numpy as np
from xgboost import XGBClassifier

from dataset import get_qmnist_numpy


def main():
    train, test = get_qmnist_numpy()
    train_X, train_y = train
    test_X, test_y = test
    train_X = train_X.reshape(train_X.shape[0], np.prod(train_X.shape[1:]))
    test_X = test_X.reshape(test_X.shape[0], np.prod(test_X.shape[1:]))
    model = XGBClassifier(device="cuda")
    model.fit(train_X, train_y, eval_set=[(test_X, test_y)], verbose=True)
    print(f"{model.score(test_X, test_y)=}")


if __name__ == "__main__":
    main()

import numpy as np
from thundersvm import SVC

import config
from dataset import get_qmnist_numpy
from results import display_results


def run():
    train, test = get_qmnist_numpy()
    train_X, train_y = train
    test_X, test_y = test
    train_X = train_X.reshape(train_X.shape[0], np.prod(train_X.shape[1:]))
    test_X = test_X.reshape(test_X.shape[0], np.prod(test_X.shape[1:]))
    model = SVC()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    return display_results(test_y, pred_y)


if __name__ == "__main__":
    app = config.get()["app"]
    app.command()(run)
    app()

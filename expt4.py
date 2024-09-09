import numpy as np
from xgboost import XGBClassifier

import config
from dataset import get_qmnist_numpy
from results import display_results


__config = config.get()["xgboost"]
__app = config.get()["app"]


@__app.command()
def main():
    run()


def run():
    train, test = get_qmnist_numpy()
    train_X, train_y = train
    test_X, test_y = test
    train_X = train_X.reshape(train_X.shape[0], np.prod(train_X.shape[1:]))
    test_X = test_X.reshape(test_X.shape[0], np.prod(test_X.shape[1:]))
    model = XGBClassifier(n_jobs=12, device="cuda")
    model.fit(train_X, train_y, eval_set=[(test_X, test_y)], verbose=True)
    pred_y = model.predict(test_X)
    display_results(test_y, pred_y)


if __name__ == "__main__":
    __app()

from sklearn import metrics
import pandas as pd
import io
import subprocess


def calculate_results(expected_y, predicted_y):
    return pd.DataFrame.from_dict(
        {
            "R²": [metrics.r2_score(expected_y, predicted_y)],
            "Accuracy": [metrics.accuracy_score(expected_y, predicted_y)],
            "RMS Error": [metrics.root_mean_squared_error(expected_y, predicted_y)],
        }
    )


def display_results(expected_y, predicted_y):
    print(f"{' Experiment Results ':=^120}")
    print(calculate_results(expected_y, predicted_y))
    cm = metrics.confusion_matrix(expected_y, predicted_y)
    cm_fig = metrics.ConfusionMatrixDisplay(cm).plot()
    display_figure(cm_fig.figure_)


def display_figure(fig):
    fig_buf = io.BytesIO()
    fig.savefig(fig_buf, format="png")
    fig_buf.seek(0)
    subprocess.Popen(["kitten", "icat"], stdin=subprocess.PIPE).communicate(
        input=fig_buf.read()
    )

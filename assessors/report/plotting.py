import colorsys
from typing import *

from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

plt.style.use('default')


def prediction_to_label(prediction) -> int:
    return np.argmax(prediction, axis=1)[0]


def lighten(rgb, scale):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb[:-1])
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale), s=s)


def plot_acc_per_class_crisp(df: pd.DataFrame, threshold: float = 0.5) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4))

    syst_acc = metrics.accuracy_score(df.inst_target, df.syst_prediction.map(prediction_to_label))
    syst_pred_acc = df.syst_prediction.map(lambda p: np.max(p, axis=1)[0] > threshold).mean()
    asss_pred_acc = df.asss_prediction.map(lambda p: p > threshold).mean()

    syst_class_accs = []  # The times the system is correct
    syst_pred_class_accs = []  # The times the system predicts itself to be correct
    asss_pred_class_accs = []  # The times the assessor predicted the system to be correct

    for target in np.sort(df.inst_target.unique()):
        selected = df.loc[df.inst_target == target]
        syst_class_accs.append(
            metrics.accuracy_score(
                selected.inst_target,
                selected.syst_prediction.map(prediction_to_label)))

        syst_pred_class_accs.append(
            selected.syst_prediction.map(lambda p: np.max(p, axis=1)[0] > threshold).mean())

        asss_pred_class_accs.append(
            selected.asss_prediction.map(lambda p: p > threshold).mean())

    labels = np.sort(df.inst_target.unique())
    x = np.arange(len(labels))
    width = 0.20

    syst_acc_bar = ax.bar(
        x - width * 1, syst_class_accs, width, label="System accuracy")
    syst_pred_acc_bar = ax.bar(
        x - width * 0.0, syst_pred_class_accs, width, label="System self predicted accuracy")
    asss_pred_acc_bar = ax.bar(
        x + width * 1, asss_pred_class_accs, width, label="Assessor predicted accuracy")

    # Draw horizontal lines for total accuracy
    corresponding_color = lambda bar: lighten(bar.patches[0].get_facecolor(), 0.8)

    def draw_hline(y, label, color):
        line = ax.axhline(y, ls="dotted", lw=2, c=color, label=f"{y:.3f}")
        return line

    l1 = draw_hline(syst_acc, f"{syst_acc}", corresponding_color(syst_acc_bar))
    l2 = draw_hline(syst_pred_acc, f"{syst_pred_acc}", corresponding_color(syst_pred_acc_bar))
    l3 = draw_hline(asss_pred_acc, f"{asss_pred_acc}", corresponding_color(asss_pred_acc_bar))

    # Fix styling of the plot
    # ax.set_title("Class Wise Aggregation")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    line_legend = plt.legend(handles=[l1, l2, l3], loc="center right")
    plt.gca().add_artist(line_legend)
    ax.legend(loc="lower right", handles=[syst_acc_bar, syst_pred_acc_bar, asss_pred_acc_bar])

    return fig


class CalibrationInfo(TypedDict):
    bins: Any


class CalibrationBin(TypedDict):
    avg_prob: float
    avg_acc: float
    count: int


def assessor_calibration_info(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    # report threshold bigger than 0
    probabilities = df.asss_prediction
    y_true = df.syst_pred_score
    y_pred = probabilities.map(lambda p: p[0] > 0.5)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probabilities, bins, right=True)

    bins = []
    for b in range(n_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bins.append({
                "avg_prob": np.mean(probabilities[selected]),
                "avg_acc": np.mean(y_true[selected] == y_pred[selected]),
                "count": len(selected)
            })

    return {
        "bins": bins,
    }


def plot_assessor_prob_histogram(df: pd.DataFrame, n_bins: int = 20, draw_averages: bool = True) -> Figure:
    # render average confidence
    # render average accuracy
    # render bin
    probabilities = df.asss_prediction

    bin_size = 1.0 / n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probabilities, bins, right=True)

    counts = np.zeros(n_bins)
    for b in range(n_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            counts[b] = len(selected)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title("Probability Histogram")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.bar(
        x=bins[:-1] + bin_size / 2.0,  # type: ignore
        height=counts,
        width=bin_size,
    )
    if draw_averages:
        avg_conf = np.mean(probabilities)
        conf_plt = ax.axvline(x=avg_conf, ls="dotted", lw=3,
                              c="#444", label="Avg. probability")
        ax.legend(handles=[conf_plt])

    return fig

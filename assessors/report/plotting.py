import colorsys
import math
from typing import *

from matplotlib.figure import Figure
from matplotlib.axes import Axes
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


def plot_acc_per_class_all(df: pd.DataFrame, crisp_threshold: Optional[float] = None) -> Figure:
    system_ids = sorted(df.syst_features.unique())
    labels = np.sort(df.inst_target.unique())

    n_systems = len(system_ids)
    n_cols = 2
    n_rows: int = math.ceil(n_systems / n_cols)

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(12, 4 * n_rows))
    for i, system_id in enumerate(system_ids):
        system_df = df.loc[df.syst_features == system_id]
        data = calc_acc_per_class(system_df, labels, crisp_threshold)
        system_ax = ax[i // n_cols, i % n_cols]
        _plot_acc_per_class(system_ax, data, labels)

    # Remove empty axes
    for i in range(n_systems, n_rows * n_cols):
        ax[i // n_cols, i % n_cols].axis('off')

    plt.tight_layout()

    return fig


def plot_acc_per_class_single(df: pd.DataFrame, crisp_threshold: Optional[float] = None) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    data = calc_acc_per_class(df, crisp_threshold)
    _plot_acc_per_class(ax, data, np.sort(df.inst_target.unique()))
    return fig


class AccPerClass(TypedDict):
    # TODO: Add Anify type generic type
    syst_acc: Union[float, Any]       # The times the system is correct
    syst_pred_acc: Union[float, Any]  # The times the system predicts itself to be correct
    asss_pred_acc: Union[float, Any]  # The times the assessor predicts the system to be correct
    support: int
    syst_class_accs: Union[List[float], Any]
    syst_pred_class_accs: List[float]
    asss_pred_class_accs: List[float]
    class_supports: List[int]


def calc_acc_per_class(df: pd.DataFrame, labels, crisp_threshold: Optional[float] = None) -> AccPerClass:
    assert len(df) > 0

    # Metrics & helpers
    conf = lambda pred: np.max(pred, axis=1)[0]
    actual_acc = lambda df: metrics.accuracy_score(
        df.inst_target, df.syst_prediction.map(prediction_to_label))
    predicted_acc_crisp = lambda preds: preds.map(lambda p: p > crisp_threshold).mean()
    predicted_acc_prob = lambda preds: preds.mean()
    predicted_acc = lambda preds: \
        predicted_acc_crisp(preds) if crisp_threshold is not None \
        else predicted_acc_prob(preds)

    # Total accuracy and predicted accuracy
    syst_acc = actual_acc(df)
    syst_pred_acc = predicted_acc(df.syst_prediction.map(conf))
    asss_pred_acc = predicted_acc(df.asss_prediction)

    # Per class accuracies and predicted accuracies
    class_dfs = [df.loc[df.inst_target == target] for target in labels]
    syst_class_accs = [actual_acc(df) for df in class_dfs]
    syst_pred_class_accs = [predicted_acc(df.syst_prediction.map(conf)) for df in class_dfs]
    asss_pred_class_accs = [predicted_acc(df.asss_prediction) for df in class_dfs]

    return {
        "syst_acc": syst_acc,
        "syst_pred_acc": syst_pred_acc,
        "asss_pred_acc": asss_pred_acc,
        "support": len(df),
        "syst_class_accs": syst_class_accs,
        "syst_pred_class_accs": syst_pred_class_accs,
        "asss_pred_class_accs": asss_pred_class_accs,
        "class_supports": df.inst_target.groupby(df.inst_target).count(),
    }


def _plot_acc_per_class(ax: Axes, data: AccPerClass, labels: List[str]):
    x = np.arange(len(labels))
    width = 0.20

    # Plot bars
    syst_acc_bar = ax.bar(
        x - width * 1,
        data['syst_class_accs'],
        width,
        label="System accuracy"
    )
    syst_pred_acc_bar = ax.bar(
        x - width * 0.0,
        data['syst_pred_class_accs'],
        width,
        label="System self predicted accuracy")
    asss_pred_acc_bar = ax.bar(
        x + width * 1,
        data['asss_pred_class_accs'],
        width,
        label="Assessor predicted accuracy")

    # Draw horizontal lines for total accuracy
    corresponding_color = lambda bar: lighten(bar.patches[0].get_facecolor(), 0.8)

    def draw_hline(y, color, with_diff=False):
        label = f"{y:.3f}"
        if with_diff:
            label += f" ({y - data['syst_acc']:+.3f})"
        line = ax.axhline(y, ls="dotted", lw=2, c=color, label=label)
        return line

    l1 = draw_hline(data['syst_acc'], corresponding_color(syst_acc_bar))
    l2 = draw_hline(data['syst_pred_acc'], corresponding_color(syst_pred_acc_bar), True)
    l3 = draw_hline(data['asss_pred_acc'], corresponding_color(asss_pred_acc_bar), True)

    # Fix styling of the plot
    # ax.set_title("Class Wise Aggregation")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n ({s})" for s, l in zip(data['class_supports'], labels)])
    # ax.set_xticklabels(labels)
    line_legend = ax.legend(handles=[l1, l2, l3], loc="lower right")
    ax.add_artist(line_legend)
    ax.legend(loc="lower left", handles=[syst_acc_bar, syst_pred_acc_bar, asss_pred_acc_bar])


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

    _bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probabilities, _bins, right=True)

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

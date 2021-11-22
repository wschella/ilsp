from abc import ABC, abstractmethod
import traceback
from typing import *

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.calibration as calibration
import pandas as pd
import numpy as np


class ResultsRefContainer():
    results: pd.DataFrame

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results


class ResultsCopyContainer():
    results: pd.DataFrame

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results.copy(deep=True)


class Component(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

    def _repr_html_(self) -> str:
        try:
            return self.render()
        except Exception as e:
            return f'''
            <div class="alert alert-danger">
                <h3>{type(self).__name__}</h3>
                <h4>{type(e).__name__}</h4>
                <pre>{e}</pre>
                <pre>{traceback.format_exc()}</pre>
            <div>
            '''

    def __str__(self) -> str:
        return self._repr_html_()


class Plot(Component):
    def __init__(self, fig: Figure) -> None:
        self.fig = fig

    def render(self) -> str:
        plt.close(fig=self.fig)
        return f'<img src="data:image/png;base64,{fig_to_base64(self.fig)}">'

# -----------------------------------------------------------------------------


class ReportResult(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        return f'''
        <h1>Report</h1>
        {DFSanityCheck(df)}
        {SystemResults(df)}
        {AssessorResults(df)}
        '''


class DFSanityCheck(Component, ResultsRefContainer):

    def render(self) -> str:
        # Surprisingly, this is not in place, so we don't need to copy
        df = self.results.drop(columns=['syst_prediction'])

        return f'''
        <div>
            <h2>Dataframe Sanity Check</h2>
            <p>
                <b>Number of rows:</b> {df.shape[0]} <br>
                <b>Example syst_prediction:</b> <br>
                {self.results.syst_prediction[1]}
            </p>
            {df.head(5).to_html()}
        </div>
        '''


class SystemResults(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        predictions = df.syst_prediction.map(prediction_to_label)

        mispredictions = (df.syst_pred_score == 0).sum()
        n_predictions = df.shape[0]

        accuracy = metrics.accuracy_score(df.inst_target, predictions)
        report = metrics.classification_report(df.inst_target, predictions)

        return f'''
        <div>
            <h2>System Results</h2>
            <p>Mispredictions: {mispredictions}/{n_predictions} ({100*accuracy:.2f}% acc)</p>
            <pre>{report}</pre>
        </div>
        '''


class AssessorResults(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        return f'''
        <div>
            <h2>Assessor Results</h2>
            {AssessorMetrics(df)}
            {AssessorConfusionMatrix(df)}
            {AssessorROCCurve(df)}
            {AssessorPrecRecallCurve(df)}
            {AssessorCalibration(df)}
        </div>
        '''


class AssessorMetrics(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results

        brier = metrics.brier_score_loss(df.syst_pred_score, df.asss_prediction)
        return f'''
        <div>
            <h3>Assessor Metrics</h3>
            <p>Brier Score: {brier}     vs    TODO<p>
        </div>
        '''


class AssessorConfusionMatrix(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        predictions = df.asss_prediction.map(threshold_prediction(0.5))
        conf_t50 = metrics.ConfusionMatrixDisplay.from_predictions(
            df.syst_pred_score,
            predictions,
            display_labels=["Failure (0)", "Succes (1)"],
            colorbar=False
        )
        return f'''
        <div>
            <h3>Assessor Confusion Matrix</h3>
            {Plot(fig=conf_t50.figure_)}
        </div>
        '''


class AssessorROCCurve(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        roc = metrics.RocCurveDisplay.from_predictions(
            y_true=df.syst_pred_score,
            y_pred=df.asss_prediction,
        )
        return f'''
        <div>
            <h3>Assessor ROC Curve</h3>
            {Plot(fig=roc.figure_)}
        </div>
        '''


class AssessorPrecRecallCurve(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        prec_recall = metrics.PrecisionRecallDisplay.from_predictions(
            y_true=df.syst_pred_score,
            y_pred=df.asss_prediction
        )
        return f'''
        <div>
            <h3>Assessor Precision-Recall Curve</h3>
            <pre>{Plot(fig=prec_recall.figure_)}</pre>
        </div>
        '''


class AssessorCalibration(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        probabilities = df.asss_prediction.map(lambda p: p[0])
        cal = calibration.CalibrationDisplay.from_predictions(
            name="assessor",
            y_true=df.syst_pred_score,
            y_prob=probabilities,
            n_bins=10,
            strategy="quantile")

        # TODO: There is a discrepancy between reporting of accuracy
        # for classification and actual positive labels for binary
        # syst_pred_true_target_prob = list(
        #     map(lambda p, t: p[0][t], df.syst_prediction, df.inst_target))
        # calibration.CalibrationDisplay.from_predictions(
        #     name="system confidence",
        #     ax=cal.ax_,
        #     n_bins=10,
        #     y_true=df.syst_pred_score,
        #     y_prob=syst_pred_true_target_prob,
        #     strategy="quantile"
        # )

        return f'''
        <div>
            <h3>Assessor Calibration</h3>
            {Plot(fig=cal.figure_)}
            {Plot(fig=plot_assessor_prob_histogram(df))}
            <div>
                {Quantiles(probabilities, n_quantiles=5)}
            </div>
        </div>
        '''


class CalibrationInfo(TypedDict):
    bins: Any


class CalibrationBin(TypedDict):
    avg_prob: float
    avg_acc: float
    count: int


def assessor_calibration_info(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
    # report threshold bigger than 0
    probabilities = df.asss_prediction.map(lambda p: p[0])
    y_true = df.syst_pred_score
    y_pred = df.asss_prediction.map(lambda p: p[0] > 0.5)

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
    probabilities = df.asss_prediction.map(lambda p: p[0])

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


class Quantiles(Component):
    """
    https://pandas.pydata.org/docs/reference/api/pandas.arrays.IntervalArray.html
    https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.html
    """

    def __init__(self, series: pd.Series, n_quantiles: int = 5) -> None:
        self.series = series
        self.n_quantiles = n_quantiles

    def render(self) -> str:
        from pandas.arrays import IntervalArray

        quant_size = len(self.series) / self.n_quantiles
        quants = pd.qcut(self.series, self.n_quantiles)
        ia: IntervalArray = quants.dtype.categories

        breakpoints = list(ia.right.values)  # type: ignore
        li = "\n".join(f'<li>{li}</li>' for li in breakpoints)
        return f'''
        <div>
            <p>
                <em>{self.n_quantiles}</em> quantiles,
                <em>{quant_size}</em> per quantile,
                <em>{len(self.series)}</em> total
            </p>
            <p>
                median: {self.series.median():.2f},
                mean:   {self.series.mean():.2f},
            </p>
            <ul>
                {li}
            </ul>
        </div>
        '''


# ---------------


def prediction_to_label(prediction) -> int:
    return np.argmax(prediction, axis=1)[0]


def threshold_prediction(threshold: float) -> Callable[[Any], int]:
    return lambda prediction: 1 if prediction[0] > threshold else 0


def fig_to_base64(fig):
    import base64
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format='png')
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded

# ----------------

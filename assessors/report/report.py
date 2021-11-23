from typing import *

import sklearn.metrics as metrics
import sklearn.calibration as calibration
import pandas as pd
import numpy as np

from assessors.report.renderer import Component
from assessors.report.components import *
import assessors.report.plotting as plotting


class ResultsRefContainer():
    """
    A simple boilerplate class that contains dataframe of assessor results by reference.
    Can just be inherited from to avoid writing __init__ every time.
    """
    results: pd.DataFrame

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results


class ResultsCopyContainer():
    """
    A simple boilerplate class that contains dataframe of assessor results by copy.
    Can just be inherited from to avoid writing __init__ every time.
    """
    results: pd.DataFrame

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results.copy(deep=True)


# -----------------------------------------------------------------------------


class AssessorReport(Component, ResultsRefContainer):
    """
    Reports various results from assessor experiments as a HTML page.
    """

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
            {Plot(fig=plotting.plot_assessor_prob_histogram(df))}
            <div>
                {Quantiles(probabilities)}
            </div>
        </div>
        '''


# ---------------


def prediction_to_label(prediction) -> int:
    return np.argmax(prediction, axis=1)[0]


def threshold_prediction(threshold: float) -> Callable[[Any], int]:
    return lambda prediction: 1 if prediction[0] > threshold else 0

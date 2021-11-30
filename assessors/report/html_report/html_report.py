from typing import *

import sklearn.metrics as metrics
import sklearn.calibration as calibration
import pandas as pd
import numpy as np

from assessors.report.html_report.renderer import Component
from assessors.report.html_report.components import *
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

    def on_error(self, e: Exception) -> str:
        raise e


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
            {AssessorClassificationReport(df)}
            {AssessorROCCurve(df)}
            {AssessorPrecRecallCurve(df)}
            {AssessorCalibration(df)}
            {AssessorPerClassFailureQuantification(df)}
        </div>
        '''


class AssessorMetrics(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results

        n_predictions = df.shape[0]

        def mispredictions(threshold: float) -> str:
            pred = df.asss_prediction.map(threshold_prediction(threshold))
            acc = metrics.accuracy_score(df.syst_pred_score, pred)
            misses = (pred != df.syst_pred_score).sum()
            return f'''
            <li>
                Mispredictions (threshold {threshold*100:.0f}%): {misses}/{n_predictions} ({100*acc:.2f}% acc)<br>
            </li>
            '''

        return f'''
        <div>
            <h3>Scalar Metrics</h3>
            <ul>
                {"".join([mispredictions(threshold) for threshold in [0.5, 0.75, 0.9]])}
            </ul>
            {AssessorBrier(df)}
            {AssessorLogloss(df)}
        </div>
        '''


class AssessorBrier(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        systems = sorted(df.syst_id.unique())
        per_system = [df.loc[df.syst_id == system] for system in systems]

        conf = lambda pred: np.max(pred, axis=1)[0]

        brier = pd.DataFrame()
        brier['system_id'] = systems
        brier['assessor'] = [metrics.brier_score_loss(df.syst_pred_score, df.asss_prediction)
                             for df in per_system]
        brier['system'] = [metrics.brier_score_loss(df.syst_pred_score, df.syst_prediction.map(conf))
                           for df in per_system]
        brier['baseline_no_refinement'] = [metrics.brier_score_loss(
            df.syst_pred_score,
            len(df) * [df.syst_pred_score.mean()])
            for df in per_system]

        brier = brier.append({
            'system_id': 'all',
            'system': metrics.brier_score_loss(df.syst_pred_score, df.syst_prediction.map(conf)),
            'assessor': metrics.brier_score_loss(df.syst_pred_score, df.asss_prediction),
            'baseline_no_refinement': metrics.brier_score_loss(
                df.syst_pred_score, len(df) * [df.syst_pred_score.mean()])
        }, ignore_index=True)

        return f'''
        <div>
            <h4>Brier Score</h4>
            {brier.to_html()}
        </div>
        '''


class AssessorLogloss(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        systems = sorted(df.syst_id.unique())
        per_system = [df.loc[df.syst_id == system] for system in systems]

        conf = lambda pred: np.max(pred, axis=1)[0]

        logloss = pd.DataFrame()
        logloss['system_id'] = systems
        logloss['assessor'] = [metrics.log_loss(df.syst_pred_score, df.asss_prediction)
                               for df in per_system]
        logloss['system'] = [metrics.log_loss(df.syst_pred_score, df.syst_prediction.map(conf))
                             for df in per_system]
        logloss['baseline_no_refinement'] = [metrics.log_loss(
            df.syst_pred_score,
            len(df) * [df.syst_pred_score.mean()])
            for df in per_system]

        logloss = logloss.append({
            'system_id': 'all',
            'system': metrics.log_loss(df.syst_pred_score, df.syst_prediction.map(conf)),
            'assessor': metrics.log_loss(df.syst_pred_score, df.asss_prediction),
            'baseline_no_refinement': metrics.log_loss(
                df.syst_pred_score, len(df) * [df.syst_pred_score.mean()])
        }, ignore_index=True)

        return f'''
        <div>
            <h4>Logloss Score</h4>
            {logloss.to_html()}
        </div>
        '''


class AssessorClassificationReport(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results

        predictions = df.asss_prediction.map(threshold_prediction(0.5))
        report = metrics.classification_report(df.syst_pred_score, predictions)
        conf_t50 = metrics.ConfusionMatrixDisplay.from_predictions(
            df.syst_pred_score,
            predictions,
            display_labels=["Failure (0)", "Succes (1)"],
            colorbar=False
        )
        return f'''
        <div>
            <h3>Classification Report</h3>
            <p><strong>Threshold 50%</strong></p>
            <pre>{report}</pre>
            {Plot(fig=conf_t50.figure_)}
        </div>
        '''


class AssessorROCCurve(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        roc = metrics.RocCurveDisplay.from_predictions(
            name="assessor",
            y_true=df.syst_pred_score,
            y_pred=df.asss_prediction,
        )
        for syst_id in sorted(df.syst_id.unique()):
            selected = df.loc[df.syst_id == syst_id]
            metrics.RocCurveDisplay.from_predictions(
                name=syst_id,
                y_true=selected.syst_pred_score,
                y_pred=selected.syst_prediction.map(lambda p: np.max(p, axis=1)[0]),
                ax=roc.ax_,
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
        probabilities = df.asss_prediction
        cal = calibration.CalibrationDisplay.from_predictions(
            name="assessor",
            y_true=df.syst_pred_score,
            y_prob=probabilities,
            n_bins=10,
            strategy="quantile")

        calibration.CalibrationDisplay.from_predictions(
            name="system (confidence calibration)",
            ax=cal.ax_,
            n_bins=10,
            y_true=df.syst_pred_score,
            y_prob=df.syst_prediction.map(max_prob),
            strategy="quantile"
        )
        cal.ax_.set_title("Assessor vs. System (confidence calibration)")

        return f'''
        <div>
            <h3>Assessor (Confidence) Calibration</h3>
            {Plot(fig=cal.figure_)}
            {Plot(fig=plotting.plot_assessor_prob_histogram(df))}
            <div>
                {Quantiles(probabilities)}
            </div>
        </div>
        '''


class AssessorPerClassFailureQuantification(Component, ResultsRefContainer):
    def render(self) -> str:
        df = self.results
        return f'''
        <div>
            <h3>Soft Per-Class Failure Quantification</h3>
            {Plot(fig=plotting.plot_failure_quantification_per_class_all(df))}

            <h3>Crisp Per-Class Failure Quantification</h3>
            {Plot(fig=plotting.plot_failure_quantification_per_class_all(df, syst_threshold=0.5, asss_threshold=0.5))}

            <h3>Mixed (soft for system, crisp for assessor) Per-Class Failure Quantification</h3>
            {Plot(fig=plotting.plot_failure_quantification_per_class_all(df, syst_threshold=None, asss_threshold=0.5))}
            <p>We report the times per-class actual accuracy of the system (actually avg of the systems)
            compared to
                the systems self predicted accuracy by considering confidence <0.5 as predicted failure
                the assessors predicted accuracy, by similarly considering confidence <0.5 as predicted failure
            </p>
            <p><strong>The system accuracy is actually averaged over all the systems in the population<strong></p>
        </div>
        '''

# ---------------


def prediction_to_label(prediction) -> int:
    return np.argmax(prediction, axis=1)[0]


def max_prob(prediction) -> float:
    return np.max(prediction, axis=1)[0]


def threshold_prediction(threshold: float) -> Callable[[Any], int]:
    return lambda prediction: 1 if prediction > threshold else 0

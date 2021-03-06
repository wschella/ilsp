import pandas as pd
import sklearn.metrics as metrics

import assessors.report as rr


def print_simple(df: pd.DataFrame) -> None:
    y_pred = df.asss_prediction.map(lambda p: p > 0.5)
    y_true = df.syst_pred_score

    print(metrics.classification_report(y_true, y_pred))
    print(metrics.confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {metrics.accuracy_score(y_true, y_pred)}")
    print(f"AUC: {metrics.roc_auc_score(y_true, df.asss_prediction)}")

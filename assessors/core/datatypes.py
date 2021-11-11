from typing import *


class PredictionRecord(TypedDict):
    """
    A single record or dataset entry to feed into assessor models.

    Will actually be a TF FeaturesDict [1] at runtime.
    But we only care about the field names and their types, the API to access
    them is the same.

    [1] https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict.
    """
    inst_index: Any
    inst_features: Any
    inst_label: Any
    syst_features: Any
    syst_prediction: Any
    syst_pred_loss: Any
    syst_pred_score: Any


class AssessorPredictionRecord(TypedDict):
    # This takes up too much space, and we usually don't need it for analysis,
    # and when we do, we should load it separately and use `inst_index` instead.
    # inst_features: Any

    inst_index: Any
    inst_label: Any
    syst_features: Any
    syst_prediction: Any
    syst_pred_loss: Any
    syst_pred_score: Any
    asss_prediction: Any
    asss_pred_loss: Any

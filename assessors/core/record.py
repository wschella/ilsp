from __future__ import annotations
from typing import *

import numpy as np


class PredictionRecord(TypedDict):
    """
    A single (untyped) record or dataset entry to feed into assessor models.
    We mostly care about the field names, and that it can be converted to a Numpy
    representation (see TypedPredictionRecord)
    """
    inst_index: Any
    inst_features: Any
    inst_target: Any
    syst_features: Any
    syst_prediction: Any
    syst_pred_loss: Any
    syst_pred_score: Any


class TypedPredictionRecord(TypedDict):
    inst_index: int
    inst_features: np.ndarray
    inst_target: np.ndarray
    syst_features: np.ndarray
    syst_prediction: np.ndarray
    syst_pred_loss: float
    syst_pred_score: float


class AssessorPredictionRecord(TypedDict):
    inst_index: int
    inst_target: np.ndarray
    # This takes up too much space, and we usually don't need it for analysis,
    # and when we do, we should load it separately and use `inst_index` instead.
    # inst_features: Any

    syst_features: np.ndarray
    syst_prediction: np.ndarray
    syst_pred_loss: float
    syst_pred_score: float
    asss_prediction: np.ndarray
    asss_pred_loss: float

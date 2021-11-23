import json

import pandas as pd
import numpy as np


def as_classification_with_binary_reward(df: pd.DataFrame) -> pd.DataFrame:
    df.syst_prediction = df.syst_prediction.map(lambda s: np.array(json.loads(s)))
    df.asss_prediction = df.asss_prediction.map(lambda s: np.array(json.loads(s)))

    df.asss_prediction = df.asss_prediction.map(lambda p: p[0])

    return df

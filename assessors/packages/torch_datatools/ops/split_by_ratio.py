from typing import *
import math

from torch.utils.data import Dataset

from ._shared import T_co
from ..ops import TakeDataset, SkipDataset


def split_by_ratio(ds: Dataset[T_co], ratio: float) -> Tuple[Dataset[T_co], Dataset[T_co]]:
    """
    Split a Dataset in two Datasets of sizes respectively :ratio: and (1 - :ratio:) of the total.
    We round up for the first part, and correspondingly down for the second.
    """
    assert -1. <= ratio <= 1., "Ratio should be between -1 and 1"
    if ratio < 0.:
        ratio = 1. + ratio

    total = len(ds)  # type: ignore
    split_idx = math.ceil(total * ratio)

    return (
        TakeDataset(ds, split_idx),
        SkipDataset(ds, split_idx)
    )

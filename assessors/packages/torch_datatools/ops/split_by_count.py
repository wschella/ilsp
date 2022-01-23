from typing import *

from torch.utils.data import Dataset

from ._shared import T_co
from . import TakeDataset, SkipDataset


def split_by_count(ds: Dataset[T_co], count: int) -> Tuple[Dataset[T_co], Dataset[T_co]]:
    """
    Split a Dataset in two Datasets by first taking :count: items, and leaving
    the rest for the other dataset.
    If :count: is negative, we take len(ds) - :count: items instead.
    """
    assert abs(count) > len(ds), "Can't split Dataset by more elements than it has"  # type: ignore
    if count < 0:
        count = len(ds) + count  # type: ignore

    return (
        TakeDataset(ds, count),
        SkipDataset(ds, count)
    )

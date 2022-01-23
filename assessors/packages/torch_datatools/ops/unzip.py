from typing import *

from torch.utils.data import Dataset

from . import TransformDataset

A = TypeVar('A')
B = TypeVar('B')


def unzip(ds: Dataset[Tuple[A, B]]) -> Tuple[Dataset[A], Dataset[B]]:
    """
    Unzip a Dataset of tuples into a tuple of Datasets
    """
    return (
        TransformDataset(ds, lambda e: e[0]),
        TransformDataset(ds, lambda e: e[1])
    )

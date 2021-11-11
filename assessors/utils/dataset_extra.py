from typing import *

from assessors.core import Dataset


def k_folds(ds: Dataset, n_folds: int) -> List[Tuple[Dataset, Dataset]]:
    """
    Create K (train, test) pairs, where the test datasets are non-overlapping
    folds, and the train sets are the complement of the corresponding test set.
    If the size of the total set can not be split cleanly, the remainder is dropped.
    """
    if n_folds <= 1:
        raise ValueError(f"The number of folds can't be 1 or lower, but is {n_folds}")

    # Size of the test folds (except the last, which might have the remainder)
    size = len(ds) // n_folds

    folds: List[Tuple[Dataset, Dataset]] = []
    for fold_i in reversed(range(n_folds)):
        test_set_start = fold_i * size
        test_set_end = test_set_start + size

        train = ds.take(test_set_start).concat(ds.skip(test_set_end))
        test = ds.skip(test_set_start).take(size)

        folds.append((train, test))

    return folds

"""
This module provides helper functions operating on Tensorflow Dataset objects.

https://www.tensorflow.org/api_docs/python/tf/data/Dataset
"""

from functools import reduce
from typing import Iterable, List, Tuple
from math import ceil, floor

import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import Any
from tensorflow_datasets.core.dataset_info import DatasetInfo


def total_size(ds: DatasetInfo) -> int:
    """
    Return the total size (i.e. number of elements) based on the DatasetInfo
    of the Dataset. This function is just reminder there is a built-in way to do
    this that is not very discoverable.
    """
    return ds.splits.total_num_examples


def enumerate_dict(ds: tf.data.Dataset, key: str = 'index') -> tf.data.Dataset:
    """
    Enumerate a Dataset by adding the index of the entry to it's FeaturesDict.
    This will error if the DS elements are any other type than dictionaries.
    """
    return ds.enumerate().map(lambda index, features: features | {key: index})


def to_supervised(ds: tf.data.Dataset, x: str, y: str) -> tf.data.Dataset:
    """
    Convert a Dataset with FeatureDict entries to one with Tuple[x, y] entries
    for use with supervised learning problems. Much like originally adding
    `as_supervised=True` to `tensorflow_dataset.load`.
    """
    return ds.map(lambda features: (features[x], features[y]))


def concatenate_all(datasets: Iterable[tf.data.Dataset]) -> tf.data.Dataset:
    """
    Concatenate all Datasets into one.
    """
    return reduce(lambda a, b: a.concatenate(b), datasets)


def split(ds: tf.data.Dataset, part: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split a Dataset in two Datasets of sizes respectively :part: and 1 - :part:
    of the total.
    We round up for the first one, and down for the second.
    """
    total = ds.cardinality()
    end = ceil(total * part)
    return (ds.take(end), ds.skip(end))


def split_absolute(ds: tf.data.Dataset, count: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split a Dataset in two Datasets by first taking :count: items, and leaving
    the rest for the other dataset.
    """
    return (ds.take(count), ds.skip(count))


def unzip(ds: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Unzip a Dataset of tuples into a tuple of Datasets
    """
    return (
        ds.map(lambda a, _b: a),
        ds.map(lambda _a, b: b)
    )


def peek(ds: tf.data.Dataset) -> Any:
    """
    Peek at the first element of a Dataset. Likely highly inefficient.
    """
    return iter(ds.take(1)).get_next()  # type: ignore


def k_folds(ds: tf.data.Dataset, n_folds: int) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    Create K (train, test) pairs, where the test datasets are non-overlapping
    folds, and the train sets are the complement of the corresponding test set.
    If the size of the total set can not be split cleanly, the remainder is dropped.
    """
    if n_folds <= 1:
        raise ValueError(f"The number of folds can't be 1 or lower, but is {n_folds}")

    # Size of the test folds (except the last, which might have the remainder)
    size = ds.cardinality() // n_folds

    folds: List[Tuple[tf.data.Dataset, tf.data.Dataset]] = []
    for fold_i in range(n_folds):
        test_set_start = fold_i * size
        test_set_end = test_set_start + size

        train = ds.take(test_set_start).concatenate(ds.skip(test_set_end))
        test = ds.skip(test_set_start).take(size)

        folds.append((train, test))

    return folds

from tensorflow.python.data.ops.dataset_ops import Dataset

import assessors.utils.dataset_extra as dse


def test_split_absolute_positive():
    ds = Dataset.range(10)
    (train, test) = dse.split_absolute(ds, 7)
    assert len(train) == 7
    assert len(test) == 3
    assert list(train) == list(range(7))
    assert list(test) == list(range(7, 10))


def test_split_absolute_negative():
    ds = Dataset.range(10)
    (train, test) = dse.split_absolute(ds, -7)
    assert len(train) == 3
    assert len(test) == 7
    assert list(train) == list(range(3))
    assert list(test) == list(range(3, 10))

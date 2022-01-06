from tensorflow.python.data.ops.dataset_ops import Dataset

import assessors.core as core
import assessors.utils.dataset_extra_tensorflow as dsetf
import assessors.utils.dataset_extra as dse


def l(ds):
    return list(ds.as_numpy_iterator())


def test_split_absolute_positive():
    ds = Dataset.range(10)
    (train, test) = dsetf.split_absolute(ds, 7)
    assert len(train) == 7
    assert len(test) == 3
    assert list(train) == list(range(7))
    assert list(test) == list(range(7, 10))


def test_split_absolute_negative():
    ds = Dataset.range(10)
    (train, test) = dsetf.split_absolute(ds, -7)
    assert len(train) == 3
    assert len(test) == 7
    assert list(train) == list(range(3))
    assert list(test) == list(range(3, 10))


def test_enumerate_dict():
    ds = Dataset.range(10)
    ds = ds.map(lambda i: {"true_index": i})
    ds = dsetf.enumerate_dict(ds)
    assert list(ds) == [{"true_index": i, "index": i} for i in range(10)]
    assert list(ds.skip(7)) == [{"true_index": i, "index": i} for i in range(7, 10)]


def test_concatenate_all():
    ds_1 = Dataset.range(10)
    ds_2 = Dataset.range(10, 20)
    ds_3 = Dataset.range(20, 30)
    ds = dsetf.concatenate_all([ds_1, ds_2, ds_3])
    assert list(ds) == list(range(30))


def test_enumerate_dict_kfold():
    ds = Dataset.range(10)
    ds = ds.map(lambda i: {"true_index": i})
    ds = dsetf.enumerate_dict(ds)
    kfold = dse.k_folds(core.TFDataset(ds), 2)
    concat = kfold[0][1].concat(kfold[1][1])
    assert len(concat) == 10

    def m(r):
        return list(map(lambda i: {"true_index": i, "index": i}, r))
    assert l(concat) == m(range(5, 10)) + m(range(0, 5))

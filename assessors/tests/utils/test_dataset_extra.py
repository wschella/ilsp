from tensorflow.python.data.ops.dataset_ops import Dataset

import assessors.core as core
import assessors.utils.dataset_extra as dse


def l(ds):
    return list(ds.as_numpy_iterator())


def test_kfold():
    ds = core.TFDataset(Dataset.range(10))
    kfold = dse.k_folds(ds, 2)
    assert len(kfold) == 2

    (train_0, test_0) = kfold[0]
    assert l(train_0) == list(range(0, 5))
    assert l(test_0) == list(range(5, 10))

    (train_1, test_1) = kfold[1]
    assert l(train_1) == list(range(5, 10))
    assert l(test_1) == list(range(0, 5))

from assessors.packages.torch_datatools.sources import SequenceDataset
from assessors.packages.torch_datatools.compositions import InterleaveDataset

Seq = SequenceDataset

l = lambda ds: [ds[i] for i in range(len(ds))]

l_neg = lambda ds: [ds[i] for i in range(-len(ds), 0)]


def test_interleave_simple():
    ds_1 = Seq([1, 2, 3])
    ds_2 = Seq([4, 5, 6])
    ds = InterleaveDataset([ds_1, ds_2])
    assert l(ds) == [1, 4, 2, 5, 3, 6]


def test_interleave_neg_index():
    ds_1 = Seq([1, 2, 3])
    ds_2 = Seq([4, 5, 6])
    ds = InterleaveDataset([ds_1, ds_2])
    assert l_neg(ds) == [1, 4, 2, 5, 3, 6]


def test_interleave_block():
    ds_1 = Seq([1, 2, 3, 4])
    ds_2 = Seq([5, 6, 7, 8])
    ds = InterleaveDataset([ds_1, ds_2], block_length=2)
    assert l(ds) == [1, 2, 5, 6, 3, 4, 7, 8]


def test_interleave_block_neg_index():
    ds_1 = Seq([1, 2, 3, 4])
    ds_2 = Seq([5, 6, 7, 8])
    ds = InterleaveDataset([ds_1, ds_2], block_length=2)
    assert l_neg(ds) == [1, 2, 5, 6, 3, 4, 7, 8]


def test_interleave_block_with_remainder():
    ds_1 = Seq([1, 2, 3, 4, 5])
    ds_2 = Seq([6, 7, 8, 9, 10])
    ds = InterleaveDataset([ds_1, ds_2], block_length=2)
    assert ds.remainder_block_length == 1
    assert ds.remainder_start == 8
    assert l(ds) == [1, 2, 6, 7, 3, 4, 8, 9, 5, 10]

    ds = InterleaveDataset([ds_1, ds_2], block_length=3)
    assert l(ds) == [1, 2, 3, 6, 7, 8, 4, 5, 9, 10]


def test_interleave_block_with_remainder_neg_index():
    ds_1 = Seq([1, 2, 3, 4, 5])
    ds_2 = Seq([6, 7, 8, 9, 10])
    ds = InterleaveDataset([ds_1, ds_2], block_length=2)
    assert ds.remainder_block_length == 1
    assert ds.remainder_start == 8
    assert l_neg(ds) == [1, 2, 6, 7, 3, 4, 8, 9, 5, 10]

    ds = InterleaveDataset([ds_1, ds_2], block_length=3)
    assert l_neg(ds) == [1, 2, 3, 6, 7, 8, 4, 5, 9, 10]

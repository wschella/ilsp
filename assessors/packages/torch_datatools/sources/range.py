from typing import *

from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)


class RangeDataset(Dataset[int]):
    """
    A Dataset constructed from a range. It has the same semantics as 
    the built-in range function.

    Examples
    --------
    >>> from torch_datatools.sources import RangeDataset
    >>> ds = RangeDataset(5)
    >>> [ds[idx] for idx in range(len(ds))]
    [0, 1, 2, 3, 4]
    >>> ds = RangeDataset(10, 0, -2)
    >>> [ds[idx] for idx in range(len(ds))]
    [10, 8, 6, 4, 2]
    """
    range

    def __init__(self, *args, **kwargs):
        self.range = range(*args, **kwargs)

    def __getitem__(self, index) -> int:
        return self.range[index]

    def __len__(self):
        return len(self.range)

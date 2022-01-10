from typing import *

from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)


class SequenceDataset(Dataset[T_co]):
    """
    A Dataset constructed from a simple Python sequence.

    Examples
    --------
    >>> from torch_datatools.sources import SequenceDataset
    >>> dataset = SequenceDataset([1, 2, 3])
    >>> dataset[0]
    1
    """
    sequence: Sequence[T_co]

    def __init__(self, sequence: Sequence[T_co]):
        self.sequence = sequence

    def __getitem__(self, index) -> T_co:
        return self.sequence[index]

    def __len__(self):
        return len(self.sequence)

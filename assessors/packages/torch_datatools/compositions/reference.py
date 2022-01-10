from typing import *

from torch.utils.data import Dataset

from ._shared import T_co


class ReferenceDataset(Dataset[T_co]):
    """
    A Dataset of references (i.e. indices) to another Dataset, can be used 
    to define another order.

    With ReferenceDataset, and pytorch's own Subset and Concat, I think you can
    do about everything you can do with a general sequence. 

    Examples
    --------
    >>> from torch_datatools.compositions import ReferenceDataset
    >>> from torch_datatools.sources import SequenceDataset
    >>> dataset = ReferenceDataset(SequenceDataset([1, 2, 3]), [1, 2, 0])
    >>> (dataset[0], dataset[1], dataset[2])
    (2, 3, 1)
    """
    references: Sequence[int]
    source: Dataset[T_co]

    def __init__(self, source: Dataset[T_co], references: Sequence[int]) -> None:
        assert len(references) == len(source)  # type: ignore
        self.references = references
        super().__init__()

    def __getitem__(self, index) -> T_co:
        idx = self.references[index]
        return self.source[idx]

    def __len__(self) -> int:
        return len(self.source)  # type: ignore

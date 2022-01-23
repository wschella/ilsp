from typing import *

from torch.utils.data import Dataset

from ._shared import T_co


class EnumerateDataset(Dataset[Tuple[int, T_co]]):
    """
    A dataset of enumerations, i.e. tuples with with an index in the first position.
    """
    source: Dataset[T_co]
    key: str

    def __init__(self, source: Dataset[T_co]) -> None:
        self.source = source
        super().__init__()

    def __getitem__(self, index) -> Tuple[int, T_co]:
        item = self.source[index]

        if index < 0:
            index = len(self) + index  # e.g. + -5

        return (index, item)

    def __len__(self):
        return len(self.source)  # type: ignore

from typing import *
from ._shared import out_of_bounds, T_co

from torch.utils.data import Dataset


class TakeDataset(Dataset[T_co]):
    count: int
    source: Dataset[T_co]

    def __init__(self, source: Dataset[T_co], count: int) -> None:
        self.source = source
        if count < 0:
            raise ValueError(".take(count) called with count < 0")
        if abs(count) > len(source):  # type: ignore
            raise ValueError(".take(count) called with count greater then the dataset size")

        self.count = count
        super().__init__()

    def __getitem__(self, index) -> T_co:
        if index >= len(self) or abs(index) > len(self):
            raise out_of_bounds(index, self)

        if index < 0:
            index = index - len(self.source) + self.count  # type: ignore # e.g. -1 - 70000 + 25

        return self.source[index]

    def __len__(self):
        return self.count

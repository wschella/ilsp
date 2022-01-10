from typing import *

from torch.utils.data import Dataset

from ._shared import T_co, out_of_bounds


class SkipDataset(Dataset[T_co]):
    num: int
    source: Dataset[T_co]

    def __init__(self, source: Dataset[T_co], num: int) -> None:
        self.source = source
        if num < 0:
            raise ValueError(".skip(num) called with num < 0")
        if abs(num) > len(source):  # type: ignore
            raise ValueError(".skip(num) called with num greater then the dataset size")

        self.num = num
        super().__init__()

    def __getitem__(self, index) -> T_co:
        if index >= len(self) or abs(index) > len(self):
            out_of_bounds(index, self)

        if index >= 0:
            index += self.num

        return self.source[index]

    def __len__(self):
        return len(self.source) - self.num  # type: ignore

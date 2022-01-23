from typing import *

from torch.utils.data import Dataset

from ._shared import T_co
from ..caches import Cache


class CachedDataset(Dataset[T_co]):
    source: Dataset[T_co]
    cache: Cache[T_co]

    def __init__(self, source: Dataset[T_co], cache: Cache[T_co]) -> None:
        self.source = source
        self.cache = cache
        super().__init__()

    def __getitem__(self, index) -> T_co:
        if index in self.cache:
            return self.cache[index]  # type: ignore
        else:
            item = self.source[index]
            self.cache[index] = item
            return item

    def __len__(self):
        return len(self.source)  # type: ignore

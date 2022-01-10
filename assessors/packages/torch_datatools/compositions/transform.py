from typing import *

from torch.utils.data import Dataset

Arg = TypeVar('Arg', contravariant=True)
Ret = TypeVar('Ret', covariant=True)


class TransformDataset(Dataset[Ret], Generic[Arg, Ret]):
    transform: Callable[[Arg], Ret]
    source: Dataset[Arg]

    def __init__(self, source: Dataset[Arg], transform: Callable[[Arg], Ret]) -> None:
        self.source = source
        setattr(self, 'transform', transform)
        super().__init__()

    def __getitem__(self, index) -> Ret:
        transform: Callable[[Arg], Ret] = getattr(self, 'transform')
        return transform(self.source[index])

    def __len__(self):
        return len(self.source)  # type: ignore

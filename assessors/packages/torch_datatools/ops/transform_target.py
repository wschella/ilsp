from typing import *

from torch.utils.data import Dataset

Input = TypeVar('Input', covariant=True)
Target = TypeVar('Target', contravariant=True)
TransTarget = TypeVar('TransTarget', covariant=True)


class TransformTargetDataset(Dataset[Tuple[Input, TransTarget, Any]], Generic[Input, Target, TransTarget]):
    """
    A dataset with a transformation applied only to the target/label of each entry,
    under the assumption that an entry is a tuple with the target as its second element.
    """
    transform: Callable[[Target], TransTarget]
    source: Dataset[Tuple[Input, Target, Any]]

    def __init__(self, source: Dataset[Tuple[Input, Target, Any]], transform: Callable[[Target], TransTarget]) -> None:
        self.source = source
        setattr(self, 'transform', transform)
        super().__init__()

    def __getitem__(self, index) -> Tuple[Input, TransTarget, Any]:
        transform: Callable[[Target], TransTarget] = getattr(self, 'transform')
        entry: Tuple[Input, Target, Any] = self.source[index]
        input, target, *rest = entry
        return input, transform(target), *tuple(rest)  # type: ignore

    def __len__(self):
        return len(self.source)  # type: ignore

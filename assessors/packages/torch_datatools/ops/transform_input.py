from typing import *

from torch.utils.data import Dataset

Input = TypeVar('Input', contravariant=True)
Output = TypeVar('Output', covariant=True)


class TransformInputDataset(Generic[Input, Output], Dataset[Tuple[Output, Any]]):
    """
    A dataset with a transformation applied only to the input/features of each entry, 
    under the assumption that an entry is a tuple with the input as first element.
    """
    transform: Callable[[Input], Output]
    source: Dataset[Tuple[Input, Any]]

    def __init__(self, source: Dataset[Tuple[Input, Any]], transform: Callable[[Input], Output]) -> None:
        self.source = source
        setattr(self, 'transform', transform)
        super().__init__()

    def __getitem__(self, index) -> Tuple[Output, Any]:
        transform: Callable[[Input], Output] = getattr(self, 'transform')
        entry: Tuple[Input, Any] = self.source[index]
        input, *rest = entry
        return transform(input), *tuple(rest)  # type: ignore

    def __len__(self):
        return len(self.source)  # type: ignore

from typing import *
import collections

from torch.utils.data import Dataset


class EnumeratedDictDataset(Dataset[Dict]):
    """
    A dataset of enumerated of dictionaries, i.e. with an index added to 
    each dictionary marking it's position in the dataset.
    """
    source: Dataset[Dict[Any, Any]]
    key: str

    def __init__(self, source: Dataset[Dict[Any, Any]], key: str) -> None:
        """
        Parameters
        ----------
        source : Dataset[Dict[Any, Any]]
            The source dataset.
        key : str
            The key to put the index at.
        """
        self.source = source
        self.key = key
        super().__init__()

    def __getitem__(self, index) -> Dict[Any, Any]:
        item = self.source[index]
        if not isinstance(item, collections.Mapping):
            raise ValueError(f"Item is not a dictionary but was expected to be so. Item: {item}.")

        if index >= 0:
            item[self.key] = index
        else:
            item[self.key] = len(self) + index  # e.g. + -5
        return item

    def __len__(self):
        return len(self.source)  # type: ignore

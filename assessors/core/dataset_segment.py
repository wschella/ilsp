from typing import *
from pathlib import Path

import torch.utils.data as td

from assessors.core.dataset import DatasetDescription
from assessors.core.dataset_torch import TorchDataset
from assessors.packages import torch_datatools


class URLCSVDatasetDescription(DatasetDescription):
    name: str
    url: str
    target_column: Optional[str] = None
    transform: Optional[Callable[..., Any]] = None
    target_transform: Optional[Callable[..., Any]] = None

    def __init__(self, name, url, target_column=None, transform=None, target_transform=None) -> None:
        self.name = name
        self.url = url
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform

        super().__init__()

    def download(self, path: Optional[Path] = None) -> None:
        self.load(path)

    def load(self, path: Optional[Path] = None) -> Dict[str, TorchDataset]:
        """
        Load all splits of the dataset in a dict.
        """
        ds = torch_datatools.fetchers.CSV_from_URL(
            self.name,
            self.url,
            path,
            self.transform,
            self.target_column,
            self.target_transform)

        return {
            'all': TorchDataset(ds)
        }

    def load_all(self, path: Optional[Path] = None) -> TorchDataset:
        """
        Load all the splits off the dataset combined together
        """
        ds: td.Dataset = td.ConcatDataset([ds.ds for ds in self.load(path).values()])
        return TorchDataset(ds)


OpenMLSegment = URLCSVDatasetDescription(  # type: ignore
    name='segment',
    url='https://www.openml.org/data/get_csv/18151937/phpyM5ND4',
)

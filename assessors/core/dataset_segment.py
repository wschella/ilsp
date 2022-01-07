
from __future__ import annotations
from typing import *
from pathlib import Path
import logging
import requests  # type: ignore

import pandas as pd
import numpy as np
import torchvision as tv
import torch.utils.data as td

from assessors.core.dataset import DatasetDescription, Dataset
from assessors.core.dataset_torch import TorchDataset


class URLCSVDatasetDescription(DatasetDescription):
    name: str
    url: str
    target_column: str
    transform: Optional[Callable[..., Any]] = None
    target_transform: Optional[Callable[..., Any]] = None
    DEFAULT_LOCATION = Path('./datasets/')

    def __init__(self, name, url, target_column, transform, target_transform) -> None:
        self.name = name
        self.url = url
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform

        super().__init__()

    def download(self, path: Optional[Path] = None) -> None:
        path = path or self.default_location() / f'{self.name}.csv'
        if path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        streaming_download(self.url, path)

    def load(self, path: Optional[Path] = None) -> Dict[str, TorchDataset]:
        """
        Load all splits of the dataset in a dict.
        """
        df = pd.read_csv(path or self.default_location() / f'{self.name}.csv')

        if self.transform:
            df[df.columns.difference(['b'])] = df[
                df.columns.difference(['b'])].apply(self.transform)

        if self.target_transform:
            df.loc[:, [self.target_column]] = df.loc[  # type: ignore
                :, [self.target_column]].apply(self.target_transform)

        return {
            'all': TorchDataset(PandasTorchDataset(df, target_column=self.target_column))
        }

    def load_all(self, path: Optional[Path] = None) -> TorchDataset:
        """
        Load all the splits off the dataset combined together
        """
        ds: td.Dataset = td.ConcatDataset([ds.ds for ds in self.load(path).values()])
        return TorchDataset(ds)

    def default_location(self) -> Path:
        # this is the directory
        return self.DEFAULT_LOCATION / self.name


OpenMLSegment = URLCSVDatasetDescription(  # type: ignore
    name='segment',
    url='https://www.openml.org/data/get_csv/18151937/phpyM5ND4',
)


def streaming_download(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


class PandasTorchDataset(td.Dataset):
    x: pd.DataFrame
    y: pd.DataFrame
    target_column: str

    def __init__(self, df: pd.DataFrame, target_column: str) -> None:
        self.target_column = target_column
        self.x = df.drop(target_column, axis=1)
        self.y = df.loc[:, [target_column]]
        super().__init__()

    def __getitem__(self, index):
        (self.x.iloc[index], self.y.iloc[index])

    def __len__(self):
        return len(self.x)

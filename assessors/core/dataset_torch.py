from __future__ import annotations
from typing import *
from pathlib import Path
from operator import itemgetter

import numpy as np
import pandas as pd

import torchvision as tv
import torch.utils.data as td

from assessors.core.dataset import DatasetDescription, Dataset

E = TypeVar('E', covariant=True)
M = TypeVar('M', covariant=True)


class TorchVisionDatasetDescription(DatasetDescription[E, 'TorchDataset']):
    def __init__(self, name: str, location: Path = None) -> None:
        self.location = location or Path(f"./datasets/{name}")
        super().__init__()

    def load_split(self, name: str, path: Optional[Path] = None) -> TorchDataset:
        ds = self.load(path).get(name)
        if ds is None:
            raise ValueError(f"Unknown split {name}")
        return ds

    def load_all(self, path: Optional[Path] = None) -> TorchDataset:
        """
        Load all the splits off the dataset combined together
        """
        ds: td.Dataset = td.ConcatDataset([ds.ds for ds in self.load(path).values()])
        return TorchDataset(ds)


class TorchMNISTDatasetDescription(TorchVisionDatasetDescription):
    def __init__(self, location: Path = None) -> None:
        super().__init__('MNIST', location)

    def download(self, path: Optional[Path] = None) -> None:
        root = str(path or self.location)
        tv.datasets.MNIST(root, download=True)
        print("Downloaded MNIST dataset")

    def load(self, path: Optional[Path] = None) -> Dict[str, TorchDataset]:
        """
        Load all splits of the dataset in a dict.
        """
        root = str(path or self.location)
        transform = tv.transforms.Compose([tv.transforms.PILToTensor()])
        kwargs = {'transform': transform, 'root': root, 'download': True}
        return {
            'train': TorchDataset(tv.datasets.MNIST(**kwargs, train=True)),
            'test': TorchDataset(tv.datasets.MNIST(**kwargs, train=False)),
        }


class TorchCIFAR10DatasetDescription(TorchVisionDatasetDescription):
    def __init__(self, location: Path = None) -> None:
        super().__init__('CIFAR10', location)

    def download(self, path: Optional[Path] = None) -> None:
        root = str(path or self.location)
        tv.datasets.CIFAR10(root, download=True)
        print("Downloaded CIFAR10 dataset")

    def load(self, path: Optional[Path] = None) -> Dict[str, TorchDataset]:
        """
        Load all splits of the dataset in a dict.
        """
        root = str(path or self.location)
        transform = tv.transforms.Compose([tv.transforms.PILToTensor()])
        kwargs = {'transform': transform, 'root': root, 'download': True}
        return {
            'train': TorchDataset(tv.datasets.CIFAR10(**kwargs, train=True)),
            'test': TorchDataset(tv.datasets.CIFAR10(**kwargs, train=False)),
        }


class TorchDataset(Dataset[E, 'TorchDataset']):
    def __init__(self, ds: td.Dataset) -> None:
        self.ds = ds
        super().__init__()

    def __iter__(self) -> Iterator[E]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def save(self, dest: Path) -> None:
        """
        Save the dataset to the given destination.
        """
        raise NotImplementedError()

    def take(self, n: int) -> Dataset[E, TorchDataset]:
        """
        Take the first n elements of the dataset.
        """
        raise NotImplementedError()

    def encode(self, val) -> Any:
        """
        Encode the given value into a format suitable for the dataset.
        """
        raise NotImplementedError()

    def skip(self, n: int) -> Dataset[E, TorchDataset]:
        """
        Skip the first n elements of the dataset.
        """
        raise NotImplementedError()

    def map(self, func: Callable[[E], M]) -> Dataset[M, TorchDataset]:
        raise NotImplementedError()

    def concat(self, other: TorchDataset) -> Dataset[E, TorchDataset]:
        """
        Concatenate two datasets.
        """
        raise NotImplementedError()

    def as_numpy(self) -> np.ndarray:
        """
        Convert the dataset to a numpy array.
        """
        raise NotImplementedError()

    def shuffle(self) -> Dataset[E, TorchDataset]:
        """
        Shuffle the dataset.
        """
        raise NotImplementedError()

    def unique(self) -> Dataset[E, TorchDataset]:
        """
        Return a dataset with all duplicate elements removed.
        """
        raise NotImplementedError()

    def enumerate_dict(self) -> Dataset[E, TorchDataset]:
        raise NotImplementedError()

    def split_absolute(self, split_at: int) -> Tuple[Dataset[E, TorchDataset], Dataset[E, TorchDataset]]:
        """
        Split the dataset at a given index.
        """
        raise NotImplementedError()

    def split_relative(self, ratio: float) -> Tuple[Dataset[E, TorchDataset], Dataset[E, TorchDataset]]:
        """
        Split the dataset at a given ratio.
        """
        raise NotImplementedError()

    def as_numpy_sequence(self) -> Sequence[E]:
        raise NotImplementedError()

    def interleave_with(self, others: List[TorchDataset], cycle_length: int, block_length: int = 1) -> Dataset[E, TorchDataset]:
        raise NotImplementedError()

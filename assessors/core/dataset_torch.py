from __future__ import annotations
from typing import *
from pathlib import Path
import logging

import numpy as np
import torchvision as tv
import torch.utils.data as td

from assessors.core.dataset import DatasetDescription, Dataset

E = TypeVar('E', covariant=True)
M = TypeVar('M', covariant=True)


class TorchVisionDatasetDescription(DatasetDescription[E, 'TorchDataset']):
    DEFAULT_LOCATION = Path('./datasets/')

    name: str
    splits: Dict[str, Callable[..., Any]]
    transform: Optional[Callable[..., Any]]

    def __init__(self, name: str, splits, transform) -> None:
        self.name = name
        self.splits = splits
        self.transform = transform
        super().__init__()

    def download(self, path: Optional[Path] = None) -> None:
        first_split = list(self.splits.values())[0]
        first_split(
            root=path or self.default_location(),
            download=True,
        )
        logging.info("Successfully downloaded dataset %s", self.name)

    def load(self, path: Optional[Path] = None) -> Dict[str, TorchDataset]:
        """
        Load all splits of the dataset in a dict.
        """
        splits: Dict[str, TorchDataset] = {}
        for split_name, split_loader in self.splits.items():
            split = split_loader(
                root=path or self.default_location(),
                transform=self.transform,
                download=True,
            )
            splits[split_name] = TorchDataset(split)
        return splits

    def load_all(self, path: Optional[Path] = None) -> TorchDataset:
        """
        Load all the splits off the dataset combined together
        """
        ds: td.Dataset = td.ConcatDataset([ds.ds for ds in self.load(path).values()])
        return TorchDataset(ds)

    ###

    def default_location(self) -> Path:
        return self.DEFAULT_LOCATION / self.name


TorchVisionMNIST = TorchVisionDatasetDescription(  # type: ignore
    name='MNIST',
    splits={
        'train': lambda **kwargs: tv.datasets.MNIST(**kwargs, train=True),
        'test': lambda **kwargs: tv.datasets.MNIST(**kwargs, train=False),
    },
    transform=tv.transforms.Compose([tv.transforms.PILToTensor()])
)

TorchVisionCIFAR10 = TorchVisionDatasetDescription(  # type: ignore
    name='CIFAR10',
    splits={
        'train': lambda **kwargs: tv.datasets.CIFAR10(**kwargs, train=True),
        'test': lambda **kwargs: tv.datasets.CIFAR10(**kwargs, train=False),
    },
    transform=tv.transforms.Compose([tv.transforms.PILToTensor()])
)


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

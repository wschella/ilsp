from __future__ import annotations
from abc import ABC, abstractmethod
from typing import *
from pathlib import Path

import numpy as np

E = TypeVar('E', covariant=True)
M = TypeVar('M', covariant=True)
SELF = TypeVar('SELF', bound='Dataset')
DS = TypeVar('DS', bound='Dataset')


class DatasetDescription(Generic[E, DS], ABC):
    """
    Description of a dataset, with functionality to download it, 
    information about common splits, and functionality to load specific splits
    and parts.
    """

    @abstractmethod
    def download(self, dest: Path = None) -> None:
        """
        Download the dataset to the given destination.
        """
        pass

    @abstractmethod
    def split(self, name: str) -> DS:
        """
        Return a Dataset for the given split.
        """
        pass

    @abstractmethod
    def load(self) -> Dict[str, DS]:
        """
        Load all splits of the dataset in a dict.
        """
        pass

    @abstractmethod
    def load_all(self) -> DS:
        """
        Load all the splits off the dataset combined together
        """
        pass


class Dataset(Generic[E, SELF], ABC, Iterable[E], Sized):
    """
    Here, a dataset is a collection of homogeneous elements which can
    be manipulated by composition.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[E]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def save(self, dest: Path) -> None:
        """
        Save the dataset to the given destination.
        """
        pass

    @abstractmethod
    def take(self, n: int) -> Dataset[E, SELF]:
        """
        Take the first n elements of the dataset.
        """
        pass

    @abstractmethod
    def encode(self, val) -> Any:
        """
        Encode the given value into a format suitable for the dataset.
        """
        pass

    @abstractmethod
    def skip(self, n: int) -> Dataset[E, SELF]:
        """
        Skip the first n elements of the dataset.
        """
        pass

    @abstractmethod
    def map(self, func: Callable[[E], M]) -> Dataset[M, SELF]:
        pass

    @abstractmethod
    def concat(self, other: SELF) -> Dataset[E, SELF]:
        """
        Concatenate two datasets.
        """
        pass

    @abstractmethod
    def as_numpy(self) -> np.ndarray:
        """
        Convert the dataset to a numpy array.
        """
        pass

    @abstractmethod
    def shuffle(self) -> Dataset[E, SELF]:
        """
        Shuffle the dataset.
        """
        pass

    @abstractmethod
    def unique(self) -> Dataset[E, SELF]:
        """
        Return a dataset with all duplicate elements removed.
        """
        pass

    @abstractmethod
    def enumerate_dict(self) -> Dataset[E, SELF]:
        pass

    @abstractmethod
    def split_absolute(self, split_at: int) -> Tuple[Dataset[E, SELF], Dataset[E, SELF]]:
        """
        Split the dataset at a given index.
        """
        pass

    @abstractmethod
    def split_relative(self, ratio: float) -> Tuple[Dataset[E, SELF], Dataset[E, SELF]]:
        """
        Split the dataset at a given ratio.
        """
        pass

    @abstractmethod
    def as_numpy_sequence(self) -> Sequence[E]:
        pass

    @abstractmethod
    def interleave_with(self, others: List[SELF], cycle_length: int, block_length: int = 1) -> Dataset[E, SELF]:
        pass

from typing import *
from operator import itemgetter
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset
from tensorflow_datasets.core.utils.read_config import ReadConfig


E = TypeVar('E', covariant=True)


class TFDatasetWrapper(Generic[E]):
    name: str
    as_supervised: bool

    def __init__(self, name: str, as_supervised: bool = True) -> None:
        self.name = name
        self.as_supervised = as_supervised
        super().__init__()

    def download(self):
        self.load()

    def load(self) -> Dict[str, TFDataset[E]]:
        return tfds.load(
            self.name,
            with_info=False,
            as_supervised=self.as_supervised,
            # split=["train", "test"],
            shuffle_files=False,  # We explicitly want control over shuffling ourselves
            # We don't want to cache at this stage, since we will build an
            # dataset pipeline on it first.
            read_config=ReadConfig(try_autocache=False),
        )

    def load_all(self) -> TFDataset[E]:
        train, test = itemgetter('train', 'test')(self.load())
        return train.concatenate(test)


MNIST: TFDatasetWrapper[Tuple[Any, int]] = TFDatasetWrapper('mnist')
CIFAR10: TFDatasetWrapper[Tuple[Any, int]] = TFDatasetWrapper('cifar10')


class CustomDataset(Generic[E]):
    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path

    def load_all(self) -> TFDataset[E]:
        return tf.data.experimental.load(str(self.path))

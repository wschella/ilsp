from __future__ import annotations
from typing import *
from pathlib import Path
from operator import itemgetter

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import ReadConfig

from assessors.core.dataset import DatasetDescription, Dataset
import assessors.utils.dataset_extra_tensorflow as dsetf

E = TypeVar('E', covariant=True)
M = TypeVar('M', covariant=True)


class TFDatasetDescription(DatasetDescription[E, 'TFDataset']):
    name: str

    def __init__(self, name: str):
        self.name = name

    def download(self, dest: Path = None) -> None:
        return super().download(dest=dest)

    def load(self) -> Dict[str, TFDataset[E]]:
        return self._load(None)

    def _load(self, split) -> Dict[str, TFDataset[E]]:
        dss: Dict[str, tf.data.Dataset] = tfds.load(
            self.name,
            split=split,
            with_info=False,
            as_supervised=True,
            shuffle_files=False,  # We explicitly want control over shuffling ourselves
            # We don't want to cache at this stage, since we will build an
            # dataset pipeline on it first.
            read_config=ReadConfig(try_autocache=False),
        )  # type: ignore
        return {s: TFDataset(ds) for s, ds in dss.items()}

    def split(self, name: str) -> TFDataset[E]:
        return self._load(name)[name]

    def load_all(self) -> TFDataset[E]:
        train, test = itemgetter('train', 'test')(self.load())
        return train.concat(test)


class CSVDatasetDescription(DatasetDescription[E, 'TFDataset']):
    name: str

    def __init__(self, name: str):
        self.name = name

    def download(self, dest: Path = None) -> None:
        raise NotImplementedError("This should be downloaded manually")

    def load(self) -> Dict[str, TFDataset[E]]:
        return self._load(None)

    def _load(self, split) -> Dict[str, TFDataset[E]]:
        path = Path("./datasets/") / self.name
        pdds = pd.read_csv(path)
        pdds['class'] = pd.Categorical(pdds['class']).codes  # type: ignore
        target = pdds.pop('class')
        ds = tf.data.Dataset.from_tensor_slices((pdds.values, target.values))
        # ds = tf.data.experimental.make_csv_dataset(str(path), batch_size=32, label_name="class")
        return {'all': TFDataset(ds)}

    def split(self, name: str) -> TFDataset[E]:
        return self._load(name)[name]

    def load_all(self) -> TFDataset[E]:
        return self.load()['all']


class TFDataset(Dataset[E, 'TFDataset']):
    ds: tf.data.Dataset

    def __init__(self, ds: tf.data.Dataset):
        self.ds = ds

    def __iter__(self) -> Iterator[E]:
        return self.ds.__iter__()

    def __len__(self) -> int:
        return len(self.ds)

    def __next__(self):
        return self.ds.__next__()  # type: ignore

    def save(self, dest: Path) -> None:
        return tf.data.experimental.save(self.ds, str(dest))

    def as_numpy_sequence(self) -> Sequence[E]:
        return self.ds.as_numpy_iterator()  # type: ignore

    def encode(self, val) -> Any:
        return tf.convert_to_tensor(val)

    def concat(self, other: TFDataset[E]) -> Dataset[E, TFDataset]:
        return TFDataset(self.ds.concatenate(other.ds))

    def shuffle(self, buffer_size: int = None, seed: int = None) -> Dataset[E, TFDataset]:
        return TFDataset(self.ds.shuffle(buffer_size, seed))

    def enumerate_dict(self) -> Dataset[E, TFDataset]:
        return TFDataset(dsetf.enumerate_dict(self.ds))

    def map(self, func: Callable[[E], M]) -> Dataset[M, TFDataset]:
        # The signature of the TF map function depends on the type of the entries dataset.
        # Tensorflow will unpack it if it's a tuple, we don't want that.
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
        wrapper = lambda a1, *args: func(a1) if not args else func((a1,) + args)  # type: ignore

        return TFDataset(self.ds.map(
            wrapper,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )

    def skip(self, n: int) -> Dataset[E, TFDataset]:
        return TFDataset(self.ds.skip(n))

    def as_numpy(self) -> np.ndarray:
        return tfds.as_dataframe(self.ds).to_numpy()

    def split_absolute(self, split_at: int) -> Tuple[Dataset[E, TFDataset], Dataset[E, TFDataset]]:
        ds1, ds2 = dsetf.split_absolute(self.ds, split_at)
        return TFDataset(ds1), TFDataset(ds2)

    def take(self, n: int) -> Dataset[E, TFDataset]:
        return TFDataset(self.ds.take(n))

    def unique(self) -> Dataset[E, TFDataset]:
        return TFDataset(self.ds.unique())

    def interleave_with(self, others: List[TFDataset], cycle_length: int, block_length: int = 1) -> Dataset[E, TFDataset]:
        sets = [self.ds] + [ds.ds for ds in others]
        ds = tf.data.Dataset.from_tensor_slices(sets)\
            .interleave(
                lambda ds: ds,
                cycle_length=cycle_length,
                block_length=block_length,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        return TFDataset(ds)


DS = TypeVar('DS', bound='Dataset')

DSTypes = Union[TFDataset, TFDataset]


class CustomDatasetDescription(DatasetDescription[E, DSTypes]):
    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__()

    def download(self, dest: Path = None) -> None:
        pass

    def split(self, name: str) -> DSTypes:
        return self.load()[name]

    def _load(self) -> DSTypes:
        if not self.path.exists():
            raise FileNotFoundError(f'{self.path} does not exist')

        if self.path.is_dir():
            return TFDataset(tf.data.experimental.load(str(self.path)))

        raise ValueError(f'{self.path} contains no known dataset format')

    def load_all(self) -> DSTypes:
        return self._load()

    def load(self) -> Dict[str, DSTypes]:
        return {'all': self._load()}

from __future__ import annotations
from typing import *
from operator import itemgetter
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset
from tensorflow_datasets.core.utils.read_config import ReadConfig


class PredictionRecord(TypedDict):
    """
    A single record or dataset entry to feed into assessor models.

    Will actually be a TF FeaturesDict [1] at runtime.
    But we only care about the field names and their types, the API to access
    them is the same.

    [1] https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict.
    """
    inst_index: Any
    inst_features: Any
    inst_label: Any
    syst_features: Any
    syst_prediction: Any
    syst_pred_loss: Any


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


class CustomDataset(Generic[E]):
    path: Path

    def __init__(self, path: Path) -> None:
        self.path = path

    def load_all(self) -> TFDataset[E]:
        return tf.data.experimental.load(str(self.path))

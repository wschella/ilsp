from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel

from assessors.utils import callbacks_extra as callbacks


class BaseModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def train(self, dataset, validation) -> BaseModel:
        pass

    @abstractmethod
    def evaluate(self, dataset):
        pass

    @abstractmethod
    def predict(self, entry):
        pass


Restore = Union[Literal["full"], Literal["checkpoint"], Literal["off"]]


class TFModel(BaseModel, ABC):
    model: Optional[KerasModel] = None
    path: Path
    restore: Restore

    def __init__(self, path: Path, restore: Restore = "full") -> None:
        self.path = path
        self.restore = restore
        super().__init__()

    def predict(self, entry):
        if self.model is None:
            raise ValueError("Called predict on model but model not loaded or trained yet.")

        return self.model.predict(entry)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model is None:
            raise ValueError("Called predict on model but model not loaded or trained yet.")

        return self.model(*args, **kwds)

    def loss(self, *args, **kwds):
        if self.model is None:
            raise ValueError("Called predict on model but model not loaded or trained yet.")

        return self.model.loss(*args, **kwds)

    def save(self, model):
        self.model = model
        self.model.save(self.path / "saved")

    def load(self):
        self.model = self.try_restore_saved()

    def try_restore_saved(self):
        if not self.restore == "full":
            print(f"Not restoring full model due to \"restore={self.restore}\" for {self.path}")
            return None

        try:
            model = tf.keras.models.load_model(self.path / "saved")
            print(f"Restored full model from {self.path}")
            return model
        except IOError as err:
            if "SavedModel file does not exist at" in str(err):
                print(f"Did not find any saved full models in {self.path}")
                return None
            else:
                raise err

    def get_checkpoint(self, model):
        dir = self.path / "checkpoints"

        epoch = tf.Variable(0)
        ckpt = tf.train.Checkpoint(
            model=model,
            optimizer=model.optimizer,
            epoch=epoch,
        )
        manager = tf.train.CheckpointManager(ckpt, dir, max_to_keep=1)

        if not self.restore:
            print(f"Training from scratch due to \"restore={self.restore}\" for {dir}")
            return manager, epoch

        latest = manager.latest_checkpoint
        if not latest:
            print(f"Training from scratch for {dir}")
            return manager, epoch

        print("Using model checkpoint {}".format(latest))
        # We add .expect_partial(), because if a previous run completed,
        # and we consequently restore from the last checkpoint, no further
        # training is need, and we don't expect to use all variables.
        ckpt.restore(latest).expect_partial()
        return manager, epoch

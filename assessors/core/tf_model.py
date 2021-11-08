from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
from pathlib import Path
import csv
import itertools

from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset
from tensorflow.keras.models import Model as KerasModel
import assessors.utils.callbacks_extra as callbacks

from assessors.core import ModelDefinition, Restore, TrainedModel


class TFModelDefinition(ModelDefinition, ABC):
    epochs: int = 10

    @abstractmethod
    def definition(self) -> KerasModel:
        raise NotImplementedError()

    @abstractmethod
    def train_pipeline(self, ds: TFDataset) -> TFDataset:
        raise NotImplementedError()

    @abstractmethod
    def test_pipeline(self, ds: TFDataset) -> TFDataset:
        """
        Test pipeline should normalize inputs similar to the training pipeline
        and set the batch size as much as the hardware allows.
        """
        raise NotImplementedError()

    def train(self, dataset, validation, restore: Restore) -> TrainedModel:
        restore.log(self.name())

        # Try first to restore an entire saved model
        if restore.should_restore_full() and (restored_model := self.try_restore_from(restore.path)) is not None:
            return restored_model

        # Try now to restore from checkpoints
        model: KerasModel = self.definition()
        epoch = tf.Variable(0)
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer, epoch=epoch)
        dir = restore.path / "checkpoints"
        ckpt_manager = tf.train.CheckpointManager(ckpt, dir, max_to_keep=1)

        if restore.should_restore_checkpoint():
            if latest := ckpt_manager.latest_checkpoint is not None:
                # We add .expect_partial(), because if a previous run completed,
                # and we consequently restore from the last checkpoint, no further
                # training is need, and we don't expect to use all variables.
                ckpt.restore(latest).expect_partial()
                print("Using model checkpoint {}".format(latest))
            else:
                print(f"Training from scratch for {dir}")

        # Actually train
        model.fit(
            self.train_pipeline(dataset),
            epochs=self.epochs,
            validation_data=self.test_pipeline(validation),
            initial_epoch=int(epoch),
            callbacks=[callbacks.EpochCheckpointManagerCallback(ckpt_manager, epoch)]
        )

        return TrainedTFModel(model, self)

    def try_restore_from(self, path: Path) -> Optional[TrainedModel]:
        if (model := self.__try_restore_from(path)) is not None:
            return TrainedTFModel(model, self)
        else:
            return None

    def __try_restore_from(self, path: Path) -> Optional[KerasModel]:
        try:
            model = tf.keras.models.load_model(path)
            print(f"Restored full model from {path}")
            return model

        except IOError as err:
            if "SavedModel file does not exist at" in str(err):
                print(f"Did not find any saved full models in {path}")
                return None
            else:
                raise err


class TrainedTFModel(TrainedModel):
    model: KerasModel
    definition: TFModelDefinition

    def __init__(self, model: KerasModel, definition: TFModelDefinition):
        self.model = model
        self.definition = definition

    def loss(self, y_true, y_pred):
        return self.model.loss(y_true, y_pred)

    def score(self, y_true, predictions):
        return self.definition.score(y_true, predictions)

    def save(self, path: Path):
        self.model.save(str(path))

    def predict(self, entry):
        return self.model(entry)

    def predict_all(self, dataset):
        ds = self.definition.test_pipeline(dataset)
        return self.model.predict(ds, verbose=1)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

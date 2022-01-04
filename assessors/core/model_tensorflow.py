from __future__ import annotations
from typing import *
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras

import assessors.utils.callbacks_extra as cbe
from assessors.core import ModelDefinition, Restore, TrainedModel
from assessors.core.dataset_tensorflow import TFDataset


class TFModelDefinition(ModelDefinition, ABC):
    def train(self, dataset: TFDataset, validation: TFDataset, restore: Restore) -> TrainedModel:
        return self._train(dataset.ds, validation.ds, restore)

    def try_restore_from(self, path: Path) -> Optional[TrainedModel]:
        try:
            model = tf.keras.models.load_model(path)
            print(f"Restored full model from {path}")
            return TrainedTFModel(model, self)

        except IOError as err:
            if "SavedModel file does not exist at" in str(err):
                print(f"Did not find any saved full models in {path}")
                return None
            else:
                raise err

    # --------------------------------------------------------------------------

    epochs: int = 10

    @abstractmethod
    def definition(self) -> keras.Model:
        raise NotImplementedError()

    @abstractmethod
    def preproces_input(self, entry) -> tf.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def train_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        raise NotImplementedError()

    @abstractmethod
    def test_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Test pipeline should normalize inputs similar to the training pipeline
        and set the batch size as much as the hardware allows.
        """
        raise NotImplementedError()

    def get_fit_kwargs(self, model: keras.Model, dataset: tf.data.Dataset, **kwargs) -> Dict:
        return {}

    def _train(self, dataset, validation, restore: Restore, **kwargs) -> TrainedModel:
        restore.log(self.name())

        # Try first to restore an entire saved model
        if restore.should_restore_full() and (restored_model := self.try_restore_from(restore.path)) is not None:
            return restored_model

        # Try now to restore from checkpoints
        model: keras.Model = self.definition()
        epoch = tf.Variable(0)
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer, epoch=epoch)
        dir = restore.path / "checkpoints"
        ckpt_manager = tf.train.CheckpointManager(ckpt, dir, max_to_keep=1)

        if restore.should_restore_checkpoint():
            if (latest := ckpt_manager.latest_checkpoint) is not None:
                # We add .expect_partial(), because if a previous run completed,
                # and we consequently restore from the last checkpoint, no further
                # training is need, and we don't expect to use all variables.
                ckpt.restore(latest).expect_partial()
                print("Using model checkpoint {}".format(latest))
            else:
                print(f"Training from scratch for {dir}")

        # Get subclass specific fit kwargs
        train_ds = self.train_pipeline(dataset)
        extra_kwargs = self.get_fit_kwargs(model, train_ds, **kwargs)
        extra_callbacks = extra_kwargs.pop("callbacks", [])
        kwargs = {**extra_kwargs, **kwargs}  # We prefer caller kwargs

        # Actually train
        model.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=self.test_pipeline(validation),
            initial_epoch=int(epoch),
            callbacks=[
                cbe.EpochCheckpointManagerCallback(ckpt_manager, epoch),
            ] + extra_callbacks,
            **kwargs,
        )

        return TrainedTFModel(model, self)


class TrainedTFModel(TrainedModel):
    model: keras.Model
    definition: TFModelDefinition

    def __init__(self, model: keras.Model, definition: TFModelDefinition):
        self.model = model
        self.definition = definition

    def loss(self, y_true, y_pred):
        return self.model.loss(y_true, y_pred)  # type: ignore

    def score(self, y_true, y_pred) -> float:
        return self.definition.score(y_true, y_pred)

    def save(self, path: Path):
        self.model.save(str(path))

    def predict(self, entry, **kwargs):
        entry = self.definition.preproces_input(entry)
        entry = tf.expand_dims(entry, axis=0)
        return self.model(entry, **kwargs)

    def predict_all(self, dataset: TFDataset, **kwargs):
        ds = self.definition.test_pipeline(dataset.ds)
        return self.model.predict(ds, verbose=1, **kwargs)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label  # type: ignore

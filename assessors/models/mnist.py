from __future__ import annotations
from typing import *
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset

from assessors.models.model import TFModel
import assessors.utils.callbacks_extra as callbacks


class MNISTDefault(TFModel):
    epochs: int = 10

    @staticmethod
    def definition():
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        return model

    def train(self, dataset, validation) -> MNISTDefault:

        # Try first to restore an entire saved model
        if (model := self.try_restore_saved()) is not None:
            self.model = model
            return self

        # Try now to restore from checkpoints
        model = self.definition()
        (ckpt_manager, epoch) = self.get_checkpoint(model)
        ckpt_callback = callbacks.EpochCheckpointManagerCallback(ckpt_manager, epoch)

        # Actually train
        model.fit(
            train_pipeline(dataset),
            epochs=self.epochs,
            validation_data=test_pipeline(validation),
            initial_epoch=int(epoch),
            callbacks=[ckpt_callback]
        )

        self.save(model)
        return self

    def evaluate(self, dataset):
        self.model.evaluate(test_pipeline(dataset))


class MNISTAssessorProbabilistic(MNISTDefault):
    epochs = 15

    @staticmethod
    def definition():
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()],
        )

        return model


def train_pipeline(ds: TFDataset):
    return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .cache()\
        .shuffle(1000)\
        .batch(128)\
        .prefetch(tf.data.experimental.AUTOTUNE)


def test_pipeline(ds: TFDataset):
    return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .cache()\
        .batch(128)\
        .prefetch(tf.data.experimental.AUTOTUNE)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

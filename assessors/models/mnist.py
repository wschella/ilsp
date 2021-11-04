from __future__ import annotations
from typing import *
from pathlib import Path
import itertools

from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.data.ops.dataset_ops import Dataset as TFDataset

from assessors.models.abstraction.tf_model import TFModelDefinition
import assessors.utils.callbacks_extra as callbacks


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


class MNISTDefault(TFModelDefinition):
    epochs: int = 10

    def name(self) -> str:
        return "mnist_default"

    @staticmethod
    def definition():
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        return model

    def train_pipeline(self, ds: TFDataset) -> TFDataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .shuffle(1000)\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, ds: TFDataset) -> TFDataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def eval_pipeline(self, ds: TFDataset) -> TFDataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .batch(512)\
            .prefetch(tf.data.experimental.AUTOTUNE)


class MNISTAssessorProbabilistic(MNISTDefault):
    epochs = 15

    @staticmethod
    def definition():
        model = keras.models.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy()],
        )

        return model

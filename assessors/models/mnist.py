from __future__ import annotations
from abc import abstractmethod
from typing import *

import tensorflow as tf
import tensorflow.keras as keras

from assessors.core import TFModelDefinition


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / tf.float32(255.), label


class MNISTDefault(TFModelDefinition):
    epochs: int = 5

    def name(self) -> str:
        return "mnist_default"

    def definition(self) -> keras.Model:
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

    def score(self, y_true, y_pred) -> float:
        return float(tf.math.argmax(y_pred, axis=1) == y_true)

    def train_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .shuffle(1000)\
            .batch(128)\
            .prefetch(tf.data.experimental.AUTOTUNE)

    def test_pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .cache()\
            .batch(256)\
            .prefetch(tf.data.experimental.AUTOTUNE)


class MNISTAssessorProbabilistic(MNISTDefault):
    epochs = 15

    def definition(self) -> keras.Model:
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

    def score(self, y_true, y_pred) -> float:
        raise NotImplementedError()
